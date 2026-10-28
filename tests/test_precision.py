"""Precision rework (0.1.15): confirmation rule for auto-detected tracks, the recall profile,
manual-track gap bridging, track splitting and the settings migration."""
import json

import pytest

from videoredact.core.model import BBox, Shape, Span, VideoTrack
from videoredact.vision.auto_detect import AutoDetectOptions, confirm_track
from videoredact.vision.tracker import TrackOptions, bridge_spans


# ---- confirmation rule -------------------------------------------------------------------
def test_precise_defaults_require_three_confident_face_hits():
    o = AutoDetectOptions()
    assert o.min_hits == 3 and o.min_track_conf == 0.7
    assert not o.head_fallback and not o.emit_person_boxes
    assert confirm_track("face", 3, 0.75, 0.2, o)
    assert confirm_track("face", 50, 0.70, 0.2, o)
    assert not confirm_track("face", 2, 0.95, 0.2, o)        # too few detections
    assert not confirm_track("face", 5, 0.65, 0.2, o)        # not confident enough on average
    assert not confirm_track("face", 10, 0.90, 0.75, o)      # taller than 60 % of the frame: not a face


def test_objects_use_the_looser_rule():
    o = AutoDetectOptions()
    # YOLOX scores real phones 0.4-0.6 (bench/results/PRECISION_PROFILE.md)
    assert confirm_track("phone", 2, 0.45, 0.1, o)
    assert not confirm_track("phone", 1, 0.9, 0.1, o)
    assert not confirm_track("phone", 4, 0.35, 0.1, o)
    assert confirm_track("screen", 2, 0.4, 0.3, o)


def test_lone_head_fallback_box_is_never_kept():
    for o in (AutoDetectOptions(), AutoDetectOptions.recall_profile()):
        assert not confirm_track("face", 1, 0.9, 0.2, o, fallback_only=True)


def test_recall_profile_keeps_everything_like_before():
    r = AutoDetectOptions.recall_profile()
    assert r.min_hits == 1 and r.min_track_conf == 0.0 and r.head_fallback and r.emit_person_boxes
    assert confirm_track("face", 1, 0.5, 0.2, r)
    assert confirm_track("face", 1, 0.5, 0.9, r)             # no size cap in the recall profile
    assert confirm_track("phone", 1, 0.35, 0.1, r)
    o = AutoDetectOptions(stride=7).apply_profile(False)
    assert o.stride == 7 and o.min_hits == 1 and o.head_fallback
    assert AutoDetectOptions(stride=7).apply_profile(True).min_hits == 3


def test_recall_profile_survives_worker_serialisation():
    from dataclasses import asdict
    d = asdict(AutoDetectOptions.recall_profile())
    assert "RECALL" not in d
    assert AutoDetectOptions(**d).min_hits == 1


# ---- manual tracking: hold + bridge ------------------------------------------------------
def _span(a, b, x=0.4):
    return Span(a, b, {a: BBox(x, 0.4, 0.1, 0.1), b: BBox(x, 0.4, 0.1, 0.1)})


def test_bridge_spans_joins_nearby_fragments_and_interpolates():
    t = VideoTrack(label="document", spans=[_span(100, 110), _span(120, 130), _span(200, 210)])
    bridge_spans(t, max_gap=30)
    assert [(s.start_frame, s.end_frame) for s in t.spans] == [(100, 130), (200, 210)]
    assert t.bbox_at(115) is not None            # the gap is now covered


def test_bridge_spans_refuses_distant_boxes_and_long_gaps():
    t = VideoTrack(label="document", spans=[_span(100, 110, x=0.1), _span(120, 130, x=0.8)])
    bridge_spans(t, max_gap=30)
    assert len(t.spans) == 2                      # same timing, but the box jumped across the frame
    t = VideoTrack(label="document", spans=[_span(100, 110), _span(150, 160)])
    bridge_spans(t, max_gap=30)
    assert len(t.spans) == 2                      # gap longer than max_gap


def test_track_options_default_to_stopping_soon_after_loss():
    o = TrackOptions()
    assert o.lost_limit_s == 8.0 and not o.search_to_end
    assert o.hold_frames >= 10 and o.bridge_gap_s >= 1.0


# ---- split --------------------------------------------------------------------------------
def test_split_at_divides_a_track_into_two_tracks():
    t = VideoTrack(label="face", spans=[Span(100, 200, {100: BBox(0.1, 0.1, 0.2, 0.2), 200: BBox(0.5, 0.5, 0.2, 0.2)})])
    tail = t.split_at(150)
    assert tail is not None and tail.id != t.id and tail.label == "face"
    assert t.first_frame() == 100 and t.last_frame() == 149
    assert tail.first_frame() == 150 and tail.last_frame() == 200
    assert t.bbox_at(149) is not None and tail.bbox_at(150) is not None
    assert t.split_at(100) is None and tail.split_at(999) is None


# ---- review keys in the main window --------------------------------------------------------------
def test_main_window_delete_undo_split_toggle_and_quick_box(monkeypatch):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from videoredact.ui.main_window import MainWindow
    from videoredact.paths import settings
    w = MainWindow()
    w._load_media("tests/data/sample.mp4")
    app.processEvents()
    p = w.project
    t = VideoTrack(label="face", spans=[Span(0, 120, {0: BBox(0.1, 0.1, 0.2, 0.2), 120: BBox(0.3, 0.3, 0.2, 0.2)})])
    p.add_video_track(t)
    w._project_changed()
    w._select("video", t.id)
    # Delete + Ctrl+Z
    w.delete_selected()
    assert p.get_track(t.id) is None
    w.undo_delete()
    assert p.get_track(t.id) is t
    # E
    w._select("video", t.id)
    w.toggle_selected()
    assert t.enabled is False
    w.toggle_selected()
    assert t.enabled is True
    # X at frame 60
    monkeypatch.setattr(w, "current_frame", lambda: 60)
    w._select("video", t.id)
    w.split_selected()
    assert len(p.video_tracks) == 2
    assert t.last_frame() == 59 and min(x.first_frame() for x in p.video_tracks if x is not t) == 60
    # quick box: a drawn box starts tracking immediately with the last label, no dialog
    calls = []
    monkeypatch.setattr(w, "_run_tracking", lambda frame, bbox, label, proto, **kw: calls.append((frame, label, proto.shape)))
    settings["quick_box"] = True
    settings["last_box_label"] = "license_plate"
    settings["last_box_shape"] = None
    w.on_box_drawn(BBox(0.2, 0.2, 0.1, 0.1))
    assert calls and calls[0][1] == "license_plate" and calls[0][2] is Shape.RECT
    # the close handler asks "jobs running, cancel?" / "save first?" in modal boxes: answer Yes
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))
    monkeypatch.setattr(w, "_confirm_discard", lambda: True)
    w.close()


# ---- settings migration ---------------------------------------------------------------------
def test_settings_migration_turns_head_fallback_off_once(tmp_path, monkeypatch):
    from videoredact import paths
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"settings_rev": 1, "detect_head": True, "language": "en"}), encoding="utf-8")
    monkeypatch.setattr(paths, "settings_path", lambda: p)
    s = paths.Settings()
    assert s["detect_head"] is False and s["detect_precise"] is True and s["settings_rev"] == 2
    # the user's later choice is respected
    s["detect_head"] = True
    s.save()
    assert paths.Settings()["detect_head"] is True
