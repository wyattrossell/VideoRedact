"""Joining a drawn box to the selected track (Phase 3 of the tracking rebuild): a box drawn in a
gap of the selected track continues it, J forces it, Ctrl+Z undoes it."""
import os
from pathlib import Path

import pytest

from videoredact.core.model import BBox, Span, VideoTrack

SAMPLE = Path("tests/data/sample.mp4")
pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="sample clip missing (run scripts/make_sample.py)")


def _window(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QMessageBox
    app = QApplication.instance() or QApplication([])
    from videoredact.ui.main_window import MainWindow
    w = MainWindow()
    w._load_media(str(SAMPLE))
    app.processEvents()
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    monkeypatch.setattr(w, "_confirm_discard", lambda: True)
    return w


def test_box_in_gap_joins_selected_track_and_undo_restores(monkeypatch):
    w = _window(monkeypatch)
    p = w.project
    t = VideoTrack(label="other", spans=[Span(0, 120, {0: BBox(0.1, 0.1, 0.2, 0.2), 120: BBox(0.3, 0.3, 0.2, 0.2)}),
                                        Span(300, 359, {300: BBox(0.5, 0.5, 0.2, 0.2)})])
    p.add_video_track(t)
    w._project_changed()
    w._select("video", t.id)
    # the tracker is replaced by a canned result so the test is fast and deterministic
    from videoredact.vision import tracker as T
    canned = VideoTrack(label="other", spans=[Span(190, 250, {190: BBox(0.4, 0.4, 0.2, 0.2), 250: BBox(0.45, 0.45, 0.2, 0.2)})])
    monkeypatch.setattr(T.ObjectTracker, "track", lambda self, *a, **k: canned)

    def run_now(name, job, done, on_partial=None, on_cancel=None):
        done(job(lambda *a: None, lambda: False, lambda *a: None))
    monkeypatch.setattr(w.jobs, "run", run_now)

    monkeypatch.setattr(w, "current_frame", lambda: 200)       # inside the gap 121..299, 80 frames from an edge
    assert w._join_target(200, BBox(0.4, 0.4, 0.2, 0.2)) is t
    assert w._join_target(200, BBox(0.4, 0.4, 0.02, 0.02)) is None    # ten times smaller: not the same object
    n_before = len(p.video_tracks)
    w.on_box_drawn(BBox(0.4, 0.4, 0.2, 0.2))
    assert len(p.video_tracks) == n_before                     # no new track
    assert t.bbox_at(220) is not None and t.bbox_at(200) is not None
    assert "joined" in t.note
    # Ctrl+Z puts the track back as it was
    w.undo_delete()
    assert t.bbox_at(220) is None and t.bbox_at(60) is not None and "joined" not in (t.note or "")
    # a box where the track already has one is not a join (it starts a new track as before)
    monkeypatch.setattr(w, "current_frame", lambda: 60)
    assert w._join_target(60, BBox(0.1, 0.1, 0.2, 0.2)) is None
    # J forces the join from anywhere
    w.join_next_box()
    assert w._join_target(60, BBox(0.8, 0.8, 0.05, 0.05)) is t
    w.close()


def test_no_selection_means_no_join(monkeypatch):
    w = _window(monkeypatch)
    assert w._join_target(10, BBox(0.1, 0.1, 0.1, 0.1)) is None
    w.join_next_box()                                          # just a status message, no crash
    assert not w.__dict__.get("_join_next")
    w.close()
