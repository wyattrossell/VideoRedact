"""Working-rule tests: track back-fill / interpolation, mask dilation, audit-log writer."""
import json

import numpy as np
import pytest

from videoredact.core import video_redact
from videoredact.core.model import BBox, LogEntry, Project, Shape, Span, VideoStyle, VideoTrack
from videoredact.vision.auto_detect import extend_span, link_fragments, _near


# ------------------------------------------------------------------ back-fill / interpolation ----
def test_extend_span_backfills_and_persists_with_hold():
    s = Span(100, 120, {100: BBox(0.1, 0.1, 0.2, 0.2), 120: BBox(0.3, 0.3, 0.2, 0.2)})
    extend_span(s, stride=5, lead_in=15, lead_out=15, last_frame=130)
    assert (s.start_frame, s.end_frame) == (85, 130)          # -15 before, clamped at the last frame after
    assert s.bbox_at(85) == BBox(0.1, 0.1, 0.2, 0.2)          # first box held over the back-fill
    assert s.bbox_at(130) == BBox(0.3, 0.3, 0.2, 0.2)         # last box held over the persistence
    s2 = Span(3, 10, {3: BBox(0, 0, .1, .1)})
    extend_span(s2, stride=10, lead_in=2, lead_out=2, last_frame=1000)
    assert (s2.start_frame, s2.end_frame) == (0, 20)          # at least one stride each way, clamped at 0


def test_span_interpolates_between_keyframes():
    s = Span(0, 10, {0: BBox(0.0, 0.0, 0.2, 0.2), 10: BBox(0.5, 0.5, 0.2, 0.2)})
    b = s.bbox_at(5)
    assert b.x == pytest.approx(0.25) and b.y == pytest.approx(0.25)
    assert s.bbox_at(11) is None                               # outside the span: nothing


def test_link_fragments_bridges_short_gap_and_interpolates():
    a = VideoTrack(label="face", spans=[Span(0, 20, {0: BBox(.10, .10, .10, .10), 20: BBox(.20, .20, .10, .10)})])
    b = VideoTrack(label="face", spans=[Span(30, 50, {30: BBox(.25, .25, .10, .10), 50: BBox(.40, .40, .10, .10)})])
    far = VideoTrack(label="face", spans=[Span(30, 50, {30: BBox(.80, .80, .10, .10)})])        # different object
    other = VideoTrack(label="screen", spans=[Span(30, 50, {30: BBox(.22, .22, .10, .10)})])    # different label
    out = link_fragments([a, b, far, other], max_gap=15)
    assert len(out) == 3
    linked = next(t for t in out if "linked" in t.note)
    assert linked.first_frame() == 0 and linked.last_frame() == 50 and len(linked.spans) == 1
    mid = linked.bbox_at(25)                                   # inside the former gap: interpolated, not None
    assert mid is not None and 0.20 < mid.x < 0.25
    # gap longer than max_gap stays split
    c = VideoTrack(label="face", spans=[Span(0, 20, {0: BBox(.1, .1, .1, .1)})])
    d = VideoTrack(label="face", spans=[Span(60, 70, {60: BBox(.1, .1, .1, .1)})])
    assert len(link_fragments([c, d], max_gap=15)) == 2


def test_near_requires_similar_size_and_distance():
    a = BBox(.1, .1, .1, .1)
    assert _near(a, BBox(.15, .15, .1, .1), gap=1, max_gap=15)
    assert not _near(a, BBox(.5, .5, .1, .1), gap=1, max_gap=15)       # too far
    assert not _near(a, BBox(.1, .1, .4, .4), gap=1, max_gap=15)       # 4x the size


# ------------------------------------------------------------------ mask dilation ----
def test_bbox_padding_dilates_and_clamps():
    b = BBox(0.5, 0.5, 0.2, 0.2)
    assert b.to_pixels(100, 100) == (50, 50, 70, 70)
    assert b.to_pixels(100, 100, pad=0.25) == (45, 45, 75, 75)         # 25 % of the box on each side
    assert BBox(0.0, 0.0, 0.2, 0.2).to_pixels(100, 100, pad=0.5) == (0, 0, 30, 30)   # clamped at the frame edge


@pytest.mark.parametrize("shape", [Shape.RECT, Shape.ELLIPSE])
def test_apply_box_pad_covers_more_pixels(shape):
    def covered(pad):
        frame = np.full((200, 200, 3), 255, dtype=np.uint8)
        video_redact.apply_box(frame, BBox(0.4, 0.4, 0.2, 0.2), VideoStyle.BLACK, shape, pad)
        return int((frame[:, :, 0] == 0).sum())
    c0, c25 = covered(0.0), covered(0.25)
    assert c25 > c0 * 1.8                                       # (1.5x side)^2 = 2.25x the area, ellipse ~pi/4 of that
    frame = np.full((200, 200, 3), 255, dtype=np.uint8)
    video_redact.apply_box(frame, BBox(0.4, 0.4, 0.2, 0.2), VideoStyle.BLACK, shape, 0.25)
    assert frame[100, 100, 0] == 0 and frame[10, 10, 0] == 255   # centre covered, far corner untouched
    if shape == Shape.RECT:
        assert frame[70, 70, 0] == 0                             # padded corner covered by the rectangle
    else:
        assert frame[70, 70, 0] == 255                           # ellipse leaves the padded box's corner


def test_default_margin_is_40_percent_each_side():
    """0.1.20: every new box is half again bigger than in 0.1.19 (margin 10 % -> 40 % of the box on each side),
    so a face stays covered through the quick pans a body camera makes. The number is one constant."""
    from videoredact import paths
    from videoredact.core.model import DEFAULT_PAD
    from videoredact.core.batch import BatchOptions
    assert DEFAULT_PAD == 0.40 == VideoTrack().pad == paths.DEFAULT_SETTINGS["video_pad"] == BatchOptions(files=[]).video_pad
    t = VideoTrack(label="face", spans=[Span(0, 5, {0: BBox(0.4, 0.4, 0.2, 0.2)})], shape=Shape.RECT)
    frame = np.full((100, 100, 3), 255, dtype=np.uint8)
    video_redact.redact_frame(frame, [t], 2, VideoStyle.BLACK)
    assert frame[32, 32, 0] == 0 and frame[67, 67, 0] == 0          # 32..68 covered: 1.8x the 40..60 box
    assert frame[31, 31, 0] == 255 and frame[68, 68, 0] == 255
    assert VideoTrack.from_dict({"id": "t1", "label": "face", "spans": [], "pad": 0.1}).pad == 0.1   # a saved project keeps its boxes


def test_track_pad_used_when_rendering():
    t = VideoTrack(label="face", spans=[Span(0, 5, {0: BBox(0.4, 0.4, 0.2, 0.2)})], pad=0.25, shape=Shape.RECT)
    frame = np.full((100, 100, 3), 255, dtype=np.uint8)
    video_redact.redact_frame(frame, [t], 2, VideoStyle.BLACK)
    assert frame[36, 36, 0] == 0 and frame[33, 33, 0] == 255      # 35..65 covered (pad), 40..60 without


# ------------------------------------------------------------------ audit-log writer ----
def test_project_log_records_author_time_and_persists(tmp_path):
    p = Project(author="Officer Example")
    before = p.modified_at
    p.log("manual_edit", "moved box 3")
    p.add_video_track(VideoTrack(label="face", spans=[Span(0, 1, {0: BBox(0, 0, .1, .1)})]))
    assert [e.action for e in p.audit_log] == ["manual_edit", "video_track_added"]
    e = p.audit_log[0]
    assert isinstance(e, LogEntry) and e.user == "Officer Example" and e.details == "moved box 3"
    assert len(e.ts) >= 19 and e.ts[4] == "-" and e.ts[10] == "T"    # ISO timestamp
    assert p.modified_at >= before
    f = tmp_path / "p.vrproj"
    p.save(f)
    raw = json.loads(f.read_text(encoding="utf-8"))
    assert raw["audit_log"][0]["action"] == "manual_edit"
    q = Project.load(f)
    assert [(x.action, x.user, x.details) for x in q.audit_log] == [(x.action, x.user, x.details) for x in p.audit_log]
    q.log("reopened", "")
    assert len(q.audit_log) == 3 and q.audit_log[-1].ts >= q.audit_log[0].ts


# ------------------------------------------------------------------ edit scope ----
def _track_moving_right():
    from videoredact.core.model import BBox, Span, VideoTrack
    t = VideoTrack(label="face")
    t.spans = [Span(0, 40, {0: BBox(0.10, 0.50, 0.10, 0.10), 20: BBox(0.30, 0.50, 0.10, 0.10), 40: BBox(0.50, 0.50, 0.10, 0.10)})]
    return t


def test_apply_edit_scopes_carry_the_same_correction():
    """A box dragged up by 0.1 and doubled in height on frame 20: 'onward' corrects frames 20 and 40
    by the same offset and scale and leaves frame 0 alone; 'back' the opposite; 'track' both;
    'frame' only sets the keyframe (the old behaviour)."""
    from videoredact.core.model import BBox
    new = BBox(0.30, 0.40, 0.10, 0.20)          # same centre x, centre y 0.55 -> 0.50 (up 0.05), h x2
    for scope, expect_changed, moved in (("frame", 0, set()), ("onward", 1, {40}), ("back", 1, {0}), ("track", 2, {0, 40})):
        t = _track_moving_right()
        n = t.apply_edit(20, new, scope)
        assert n == expect_changed, scope
        kf = t.spans[0].keyframes
        assert kf[20].to_list() == new.to_list()
        for k in (0, 40):
            b = kf[k]
            if k in moved:
                assert abs(b.h - 0.20) < 1e-6 and abs(b.w - 0.10) < 1e-6, (scope, k)
                assert abs(b.center()[1] - 0.50) < 1e-6 and abs(b.center()[0] - (0.15 if k == 0 else 0.55)) < 1e-6, (scope, k)
            else:
                assert b.to_list() == [0.1 if k == 0 else 0.5, 0.5, 0.1, 0.1], (scope, k)
    # interpolated frames between an untouched and a corrected keyframe ramp between the two
    t = _track_moving_right()
    t.apply_edit(20, new, "onward")
    mid = t.bbox_at(10)
    assert 0.10 < mid.h < 0.20


def test_apply_edit_on_a_gap_frame_extends_and_corrects_nothing_else():
    from videoredact.core.model import BBox
    t = _track_moving_right()
    n = t.apply_edit(60, BBox(0.6, 0.5, 0.1, 0.1), "onward")
    assert n == 0 and t.bbox_at(60) is not None and t.spans[0].keyframes[40].to_list() == [0.5, 0.5, 0.1, 0.1]
