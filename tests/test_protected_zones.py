"""Protected zones: a region (the body camera's timestamp overlay) that is never redacted,
whatever tracks cover it; auto-detect ignores detections inside it; export keeps it visible."""
import os
from pathlib import Path

import numpy as np
import pytest

from videoredact.core import video_redact
from videoredact.core.model import (AXON_OVERLAY_ZONE, BBox, MediaInfo, Project, ProtectedZone, Span, VideoTrack,
                                    default_overlay_zone)
from videoredact.vision.auto_detect import drop_in_zones
from videoredact.vision.detectors.base import Detection

SAMPLE = Path("tests/data/sample.mp4")


def test_zone_roundtrip_and_default(tmp_path):
    p = Project(media=MediaInfo(path="x.mp4", fps=30, frame_count=30, has_video=True))
    z = p.add_protected_zone(default_overlay_zone())
    assert z.bbox == AXON_OVERLAY_ZONE and p.active_zones() == [AXON_OVERLAY_ZONE]
    assert p.audit_log[-1].action == "protected_zone_added"
    z.enabled = False
    assert p.active_zones() == []
    f = tmp_path / "z.vrproj"
    p.save(f)
    q = Project.load(f)
    assert len(q.protected_zones) == 1 and q.protected_zones[0].id == z.id and not q.protected_zones[0].enabled
    assert q.protected_zones[0].note.startswith("sized for the Axon")
    # a project saved before zones existed loads with none
    import json
    d = p.to_dict()
    del d["protected_zones"]
    f.write_text(json.dumps(d), encoding="utf-8")
    assert Project.load(f).protected_zones == []


def test_redact_frame_restores_zone_pixels():
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 255, (72, 128, 3), dtype=np.uint8)
    orig = frame.copy()
    # one black box over the whole top half; a zone inside it
    t = VideoTrack(label="face", spans=[Span(0, 9, {0: BBox(0.0, 0.0, 1.0, 0.5)})], pad=0.0)
    zone = BBox(0.5, 0.0, 0.25, 0.25)       # x 64..96, y 0..18
    out = video_redact.redact_frame(frame, [t], 3, "black", in_place=False, protected=[zone])
    assert np.array_equal(out[0:18, 64:96], orig[0:18, 64:96])          # zone untouched
    assert out[0:18, 0:64].max() == 0 and out[20:36, 64:96].max() == 0  # the rest of the box is black
    assert np.array_equal(frame, orig)                                   # in_place=False left the input alone
    video_redact.redact_frame(frame, [t], 3, "black", protected=[zone])
    assert np.array_equal(frame[0:18, 64:96], orig[0:18, 64:96]) and frame[0:18, 0:64].max() == 0
    # without zones the box is solid
    assert video_redact.redact_frame(orig.copy(), [t], 3, "black")[0:18, 64:96].max() == 0


def test_auto_detect_drops_detections_inside_zones():
    zone = AXON_OVERLAY_ZONE
    overlay_face = Detection(BBox(0.70, 0.02, 0.08, 0.08), "face", 0.8)        # inside the overlay
    half_in = Detection(BBox(0.60, 0.08, 0.10, 0.10), "face", 0.8)             # straddles the zone edge
    real = Detection(BBox(0.40, 0.40, 0.10, 0.15), "face", 0.9)
    kept = drop_in_zones([overlay_face, half_in, real], [zone])
    assert kept[-1] is real and overlay_face not in kept
    assert drop_in_zones([overlay_face], []) == [overlay_face]


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample clip missing (run scripts/make_sample.py)")
def test_export_keeps_the_zone_visible(tmp_path):
    from videoredact.core.export import ExportOptions, export_project
    from videoredact.core.media import FrameReader, probe
    proj = Project(media=probe(str(SAMPLE), compute_hash=True), author="test")
    proj.add_video_track(VideoTrack(label="screen", spans=[Span(0, 200, {0: BBox(0.0, 0.0, 1.0, 1.0)})], pad=0.0))
    z = proj.add_protected_zone(default_overlay_zone())
    out = tmp_path / "zone.mp4"
    res = export_project(proj, ExportOptions(str(out), end_s=0.5, write_report=True, write_project=False, write_audit=True))
    with FrameReader(str(SAMPLE)) as src, FrameReader(str(out)) as dst:
        W, H = src.width, src.height
        a, b = src.read_at(3), dst.read_at(3)
    x0, y0, x1, y1 = z.bbox.to_pixels(W, H)
    diff_zone = np.abs(a[y0:y1, x0:x1].astype(int) - b[y0:y1, x0:x1].astype(int)).mean()
    assert diff_zone < 12, diff_zone                       # the overlay region survives the lossy encode
    assert b[y1 + 8:, : x0 - 8].mean() < 8                 # everything else is the black box
    import json
    m = json.loads(Path(res["audit_json"]).read_text(encoding="utf-8"))
    assert m["redactions"]["protected_zones"][0]["id"] == z.id and m["redactions"]["protected_zones"][0]["enabled"] is True
    csv = Path(res["report_csv"]).read_text(encoding="utf-8")
    assert "protected_zone" in csv and "never redacted" in csv


def test_main_window_zone_toggle_and_edit():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from videoredact.ui.main_window import MainWindow
    from videoredact.ui.video_view import ZoneItem
    from videoredact.paths import settings
    settings["protect_overlay"] = True
    w = MainWindow()
    w._load_media(str(SAMPLE))
    app.processEvents()
    p = w.project
    assert len(p.protected_zones) == 1 and w.a_protect.isChecked()
    assert any(isinstance(b, ZoneItem) for b in w.video._boxes)
    z = p.protected_zones[0]
    w.on_zone_edited(z.id, BBox(0.5, 0.0, 0.4, 0.1))
    assert z.bbox.to_list() == [0.5, 0.0, 0.4, 0.1] and p.audit_log[-1].action == "protected_zone_edited"
    w.a_protect.trigger()                      # toggles off
    app.processEvents()
    assert not p.active_zones() and not w.a_protect.isChecked() and settings["protect_overlay"] is False
    assert not any(isinstance(b, ZoneItem) for b in w.video._boxes)
    w.a_protect.trigger()                      # and back on
    assert p.active_zones() and settings["protect_overlay"] is True
    w.reset_protect()
    assert p.protected_zones[0].bbox == AXON_OVERLAY_ZONE
    w.jobs.cancel_all()
    w.jobs.wait_all(10000)
    app.processEvents()                                   # lets the jobs panel drop finished jobs
    w.dirty = False
    w.close()
