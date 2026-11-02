"""Whole-picture redaction (Wyatt, 2026-11-02): one track that covers the entire picture over a
time range, for footage recorded where everything in view must be withheld."""
import os

import numpy as np
import pytest

from videoredact.core import video_redact
from videoredact.core.model import BBox, FULL_FRAME_LABEL, MediaInfo, Project, Span, VideoStyle, VideoTrack, full_frame_track


def test_full_frame_track_covers_everything_in_its_range_only():
    t = full_frame_track(30, 10, style=VideoStyle.BLUE)        # reversed range is accepted
    assert t.is_full_frame and t.label == FULL_FRAME_LABEL and t.pad == 0 and t.style is VideoStyle.BLUE
    assert (t.first_frame(), t.last_frame()) == (10, 30)
    for f in (10, 20, 30):
        assert t.bbox_at(f) == BBox(0, 0, 1, 1)
    assert t.bbox_at(9) is None and t.bbox_at(31) is None
    assert not VideoTrack(label="face").is_full_frame
    # survives a save / load round trip
    p = Project(media=MediaInfo(path="x", fps=30, frame_count=60, has_video=True))
    p.add_video_track(t)
    q = Project.from_dict(p.to_dict())
    assert q.video_tracks[0].is_full_frame and q.video_tracks[0].bbox_at(15) == BBox(0, 0, 1, 1)


def test_render_blacks_out_the_whole_frame_and_blur_leaves_no_detail():
    rng = np.random.default_rng(1)
    src = rng.integers(0, 255, (72, 128, 3), dtype=np.uint8)
    t = full_frame_track(5, 15)
    out = video_redact.redact_frame(src, [t], 10, VideoStyle.BLACK, in_place=False)
    assert out.max() == 0
    assert np.array_equal(video_redact.redact_frame(src, [t], 4, VideoStyle.BLACK, in_place=False), src)
    blurred = video_redact.redact_frame(src, [t], 10, VideoStyle.BLUR, in_place=False)
    assert blurred.std() < src.std() / 4           # a few colour patches, no texture


@pytest.mark.skipif(not os.path.exists("tests/data/sample.mp4"), reason="sample clip missing")
def test_export_is_solid_edge_to_edge_for_the_range(tmp_path):
    from videoredact.core.export import ExportOptions, export_project
    from videoredact.core.media import FrameReader, probe
    info = probe("tests/data/sample.mp4")
    proj = Project(media=info)
    proj.add_video_track(full_frame_track(10, 20))
    out = tmp_path / "full.mp4"
    export_project(proj, ExportOptions(str(out), write_report=False, write_project=False, write_audit=False))
    with FrameReader(str(out)) as r, FrameReader("tests/data/sample.mp4") as s:
        r.seek(15)
        inside = r.read()
        r.seek(40)
        after = r.read()
        s.seek(40)
        src_after = s.read()
    assert inside.max() < 24                                    # black edge to edge (lossy encode)
    assert np.abs(after.astype(int) - src_after.astype(int)).mean() < 12   # untouched outside the range


def test_main_window_adds_and_protects_the_whole_picture_track():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from videoredact.ui.main_window import MainWindow
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    w._load_media("tests/data/sample.mp4")
    app.processEvents()
    p = w.project
    t = w.add_full_frame(1.0, 2.0, VideoStyle.BLACK)
    assert t.is_full_frame and p.video_tracks[-1] is t and p.audit_log[-1].action == "full_frame_added"
    fps = w.fps
    assert t.first_frame() == round(1.0 * fps) and t.last_frame() == round(2.0 * fps) - 1
    # shown as a fixed box: no handles, no drag; a drawn box never "joins" it; T does nothing
    w.player.pause()
    w.seek(1.5)
    app.processEvents()
    items = [b for b in w.video._boxes if b.track is t]
    assert items and items[0].fixed
    w._select("video", t.id)
    assert w._join_target(w.current_frame() + 100, BBox(.1, .1, .1, .1)) is None
    w.on_box_edited(t.id, BBox(.2, .2, .2, .2))
    assert t.bbox_at(t.first_frame()) == BBox(0, 0, 1, 1)
    w.track_id(t.id)
    assert t.bbox_at(t.first_frame()) == BBox(0, 0, 1, 1) and len(p.video_tracks) == 1
    w.jobs.cancel_all()
    w.jobs.wait_all(10000)
    app.processEvents()
    w.dirty = False
    w.close()
