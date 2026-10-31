"""bench/score_project.py: scoring a candidate redaction against a hand-made reference."""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from videoredact.core.media import FrameReader, probe
from videoredact.core.model import BBox, Project, Span, VideoTrack

SAMPLE = Path("tests/data/sample.mp4")
pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="sample clip missing (run scripts/make_sample.py)")


def _load():
    spec = importlib.util.spec_from_file_location("score_project", "bench/score_project.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _reference():
    """The red square's true boxes every 5 frames, one span per presence on screen."""
    p = Project(media=probe(str(SAMPLE)), author="test")
    spans, cur = [], None
    with FrameReader(SAMPLE) as r:
        for f in range(0, 360, 5):
            img = r.read_at(f)
            m = (img[:, :, 2] > 180) & (img[:, :, 1] < 60) & (img[:, :, 0] < 60)
            ys, xs = np.where(m)
            if len(xs) > 300:
                b = BBox(xs.min() / 640, ys.min() / 360, (xs.max() - xs.min()) / 640, (ys.max() - ys.min()) / 360)
                if cur is None:
                    cur = Span(f, f, {f: b})
                    spans.append(cur)
                else:
                    cur.keyframes[f] = b
                    cur.end_frame = f
            else:
                cur = None
    p.add_video_track(VideoTrack(label="other", spans=spans))
    return p


def test_score_finds_gap_and_over_redaction(tmp_path):
    sp = _load()
    ref = _reference()
    assert len(ref.video_tracks[0].spans) == 2
    W, H = ref.media.width, ref.media.height
    # perfect candidate: the reference itself
    res = sp.score(ref, ref.video_tracks, step=3, cover=0.85, W=W, H=H)
    assert res["recall"] == 1.0 and not res["leak_runs"] and res["over_frames"] == 0
    # a candidate missing frames 60-105 and with a box on nothing
    t = ref.video_tracks[0]
    s0, s1 = t.spans
    cand = VideoTrack(label="other", spans=[Span(s0.start_frame, 59, {k: v for k, v in s0.keyframes.items() if k <= 59}), s1])
    extra = VideoTrack(label="other", spans=[Span(200, 260, {200: BBox(0.7, 0.05, 0.1, 0.1)})])
    res = sp.score(ref, [cand, extra], step=3, cover=0.85, W=W, H=H)
    assert 0.6 < res["recall"] < 0.9
    assert res["leak_runs"] and res["leak_runs"][0]["start"] >= 60 and res["leak_runs"][0]["end"] <= 105
    assert res["over_frames"] == 0           # the extra box shares frames with a covered reference box
    assert res["extra_area_mean"] > 0
    # the retrack mode: one box per reference track, tracked like the app. Scored on one continuous
    # presence of the square (whether a flat object is re-acquired across its 2 s absence depends on
    # a pixel of the drawn box; that is the engines' business, not this tool's)
    one = Project(media=ref.media, author="test")
    one.add_video_track(VideoTrack(label="other", spans=[s1]))
    tracks = sp.retrack(one, str(SAMPLE), "anchored", progress=lambda *a: None)
    assert len(tracks) == 1
    res = sp.score(one, tracks, step=3, cover=0.85, W=W, H=H)
    assert res["recall"] >= 0.9, res
    # the project round-trips through disk for the CLI
    ref.save(tmp_path / "ref.vrproj")
    assert Project.load(tmp_path / "ref.vrproj").video_tracks[0].total_frames() == t.total_frames()
