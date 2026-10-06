import json
from pathlib import Path

import numpy as np
import pytest

from videoredact.core.model import (AudioRedaction, AudioStyle, BBox, Project, Segment, Span,
                                    VideoStyle, VideoTrack, Word, MediaInfo)
from videoredact.core import audio_redact, video_redact
from videoredact.audio.transcribe import find_matches, normalize_token
from videoredact.audio.pii import suggest_pii

DATA = Path(__file__).parent / "data"


def seg(text, t0=0.0, step=0.4):
    words = []
    t = t0
    for tok in text.split():
        words.append(Word(t, t + step * 0.9, tok))
        t += step
    return Segment(t0, t, text, words)


# ---------------------------------------------------------------- model ----
def test_bbox_pixels_and_iou():
    b = BBox(0.25, 0.25, 0.5, 0.5)
    assert b.to_pixels(100, 100) == (25, 25, 75, 75)
    assert b.to_pixels(100, 100, pad=0.1) == (20, 20, 80, 80)
    assert abs(b.iou(BBox(0.25, 0.25, 0.5, 0.5)) - 1.0) < 1e-9
    assert b.iou(BBox(0.8, 0.8, 0.1, 0.1)) == 0.0
    assert BBox.from_pixels(-10, 0, 50, 50, 100, 100).x == 0.0


def test_span_interpolation_and_hold():
    s = Span(0, 10, {0: BBox(0, 0, .1, .1), 10: BBox(.5, .5, .1, .1)})
    mid = s.bbox_at(5)
    assert abs(mid.x - 0.25) < 1e-9 and abs(mid.y - 0.25) < 1e-9
    assert s.bbox_at(11) is None
    s2 = Span(0, 20, {5: BBox(.2, .2, .1, .1)})
    assert s2.bbox_at(0).x == .2 and s2.bbox_at(20).x == .2


def test_span_simplify_keeps_corners():
    kf = {i: BBox(i / 100, 0.1, .1, .1) for i in range(0, 50)}
    kf.update({i: BBox(0.49, 0.1 + (i - 49) / 100, .1, .1) for i in range(50, 100)})
    s = Span(0, 99, kf)
    s.simplify()
    assert len(s.keyframes) <= 6
    assert abs(s.bbox_at(25).x - 0.25) < 0.005
    assert abs(s.bbox_at(75).y - (0.1 + 26 / 100)) < 0.005


def test_track_merge_spans():
    t = VideoTrack(spans=[Span(0, 5, {0: BBox(0, 0, .1, .1)}), Span(6, 9, {9: BBox(0, 0, .1, .1)}),
                          Span(20, 30, {20: BBox(0, 0, .1, .1)})])
    t.merge_spans()
    assert [(s.start_frame, s.end_frame) for s in t.spans] == [(0, 9), (20, 30)]
    assert t.bbox_at(15) is None and t.bbox_at(25) is not None


def test_project_roundtrip(tmp_path):
    p = Project(media=MediaInfo(path="x.mp4", fps=30, frame_count=300, has_video=True), author="tester")
    p.transcript = [seg("hello world")]
    p.add_audio_redaction(AudioRedaction(1.0, 2.0, "world", style=AudioStyle.SILENCE))
    t = VideoTrack(label="face", style=VideoStyle.BLUR)
    t.add_keyframe(10, BBox(.1, .1, .2, .2))
    p.add_video_track(t)
    f = tmp_path / "p.vrproj"
    p.save(f)
    q = Project.load(f)
    assert q.author == "tester" and q.audio_redactions[0].style == AudioStyle.SILENCE
    assert q.video_tracks[0].bbox_at(10).w == pytest.approx(.2, abs=1e-4)
    assert len(q.audit_log) == 2
    assert json.loads(f.read_text())["format"] == "videoredact-project"


# ---------------------------------------------------------------- audio ----
def test_audio_redaction_overwrites_interior():
    sr = 8000
    x = np.random.default_rng(0).standard_normal((sr * 3, 2)).astype(np.float32) * 0.1
    r = AudioRedaction(1.0, 2.0)
    y = audio_redact.apply_redactions(x, sr, [r], AudioStyle.SILENCE)
    assert np.all(y[sr + 10: 2 * sr - 10] == 0)
    assert np.array_equal(y[: sr - 100], x[: sr - 100])
    assert np.array_equal(y[2 * sr + 100:], x[2 * sr + 100:])
    y2 = audio_redact.apply_redactions(x, sr, [r], AudioStyle.BEEP, beep_freq=1000)
    seg_ = y2[sr + 100: 2 * sr - 100, 0]
    # a 1 kHz tone at 8 kHz -> strong spectral peak at bin for 1000 Hz
    spec = np.abs(np.fft.rfft(seg_))
    peak_hz = np.argmax(spec) * sr / len(seg_)
    assert abs(peak_hz - 1000) < 5
    assert not np.any(np.isnan(y2))
    # disabled redactions are no-ops
    r.enabled = False
    assert np.array_equal(audio_redact.apply_redactions(x, sr, [r]), x)


def test_merge_ranges():
    assert audio_redact.merge_ranges([(0, 1), (0.5, 2), (3, 4)]) == [(0, 2), (3, 4)]


# ---------------------------------------------------------------- video ----
@pytest.mark.parametrize("style", list(VideoStyle))
def test_video_styles_change_region_only(style):
    frame = np.random.default_rng(1).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    orig = frame.copy()
    video_redact.apply_box(frame, BBox(0.25, 0.25, 0.5, 0.5), style)
    assert not np.array_equal(frame[30:90, 40:120], orig[30:90, 40:120])
    assert np.array_equal(frame[:30], orig[:30]) and np.array_equal(frame[:, :40], orig[:, :40])
    if style == VideoStyle.BLACK:
        assert frame[30:90, 40:120].max() == 0


def test_redact_frame_uses_track_style_or_default():
    frame = np.full((100, 100, 3), 200, dtype=np.uint8)
    t = VideoTrack(spans=[Span(0, 10, {0: BBox(0, 0, .5, .5)})], pad=0.0)
    video_redact.redact_frame(frame, [t], 5, VideoStyle.BLACK)
    assert frame[10, 10].max() == 0 and frame[90, 90].min() == 200
    assert video_redact.redact_frame(frame, [t], 50, VideoStyle.BLACK)[10, 10].max() == 0  # outside span: unchanged


# ---------------------------------------------------------- transcript -----
def test_word_matching_normalizes():
    assert normalize_token("Carter,") == "carter" and normalize_token("don't") == "don't"
    s = [seg("Michael Carter said hi."), seg("Yes, michael CARTER!", t0=5)]
    hits = find_matches(s, "Michael Carter")
    assert len(hits) == 2 and hits[1][0].start == pytest.approx(5.4)
    assert len(find_matches(s, "carter")) == 2
    assert find_matches(s, "nobody") == []


def test_pii_suggestions():
    s = [seg("My name is Michael Carter and my phone number is five five five two one two four four six seven. "
             "I live at one two three Maple Street. Date of birth March fourth nineteen ninety.")]
    kinds = {x.kind: x.text for x in suggest_pii(s)}
    assert "Michael Carter" in kinds.get("name", "")
    assert "five" in kinds.get("phone", "")
    assert "Maple Street" in kinds.get("address", "")
    assert "March" in kinds.get("dob", "")
