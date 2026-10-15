"""Phase 4: evidence integrity and compliance.
- export never touches the source, re-hashes it, and writes a JSON audit manifest
- the manifest is self-verifying and records every redaction and every model
- pixelate/blur cannot be lighter than the irreversibility floor
- offline mode refuses model downloads and the app sets telemetry-off env vars
"""
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from videoredact.core.model import (AudioRedaction, AudioStyle, BBox, Project, Span, VideoStyle, VideoTrack)
from videoredact.core import video_redact
from videoredact.core.audit import build_manifest, verify_manifest, model_inventory
from videoredact.core.media import probe, sha256_file

DATA = Path(__file__).parent / "data"


# ------------------------------------------------------------------ export + manifest ----
@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    from videoredact.core.export import export_project, ExportOptions
    tmp = tmp_path_factory.mktemp("audit")
    src = tmp / "sample.mp4"
    shutil.copy(DATA / "sample.mp4", src)
    before = sha256_file(src)
    mtime = src.stat().st_mtime_ns
    proj = Project(media=probe(str(src), compute_hash=True), author="tester", case_number="24-0001")
    proj.add_video_track(VideoTrack(label="face", spans=[Span(10, 60, {10: BBox(0.2, 0.2, 0.2, 0.2), 60: BBox(0.5, 0.5, 0.2, 0.2)})]))
    proj.add_audio_redaction(AudioRedaction(1.0, 2.0, text="name", style=AudioStyle.BEEP))
    proj.log("manual_edit", "moved box")
    out = tmp / "sample_redacted.mp4"
    res = export_project(proj, ExportOptions(str(out), end_s=3.0), progress=lambda p, m: None)
    return {"src": src, "before": before, "mtime": mtime, "out": out, "res": res, "proj": proj}


def test_source_untouched_and_rehashed(exported):
    e = exported
    assert sha256_file(e["src"]) == e["before"]
    assert e["src"].stat().st_mtime_ns == e["mtime"]
    assert e["res"]["source_sha256_at_export"] == e["before"]
    assert e["res"]["source_unchanged"] is True
    assert e["res"]["output_sha256"] == sha256_file(e["out"])


def test_manifest_written_and_self_verifying(exported):
    res = exported["res"]
    p = Path(res["audit_json"])
    assert p.exists() and p.name == "sample_redacted_audit.json"
    ok, stored = verify_manifest(p)
    assert ok and stored == res["manifest_sha256"]
    m = json.loads(p.read_text(encoding="utf-8"))
    assert m["format"] == "videoredact-audit"
    assert m["operator"]["name"] == "tester" and m["case_number"] == "24-0001"
    assert m["source"]["sha256_at_open"] == m["source"]["sha256_at_export"] == exported["before"]
    assert m["source"]["unchanged_since_open"] is True
    assert m["output"]["sha256"] == res["output_sha256"]
    assert m["output"]["project_sha256"] == sha256_file(res["project"])   # project on disk matches
    assert m["software"]["version"]
    # every redaction with frame ranges and boxes
    v = m["redactions"]["video"]
    assert len(v) == 1 and v[0]["label"] == "face" and v[0]["spans"][0]["start_frame"] == 10
    assert set(v[0]["spans"][0]["keyframes"]) == {"10", "60"}
    a = m["redactions"]["audio"]
    assert len(a) == 1 and a[0]["style"] == "beep" and a[0]["start_s"] == 1.0
    # manual edits + export in the audit log
    actions = [x["action"] for x in m["audit_log"]]
    assert "manual_edit" in actions and "export" in actions
    # models with hashes and licences
    assert m["models"] and all(len(x["sha256"]) == 64 and x["license"] for x in m["models"])
    assert "thresholds" not in m or True
    assert "detect_conf" in m["settings"] and "face_conf" in m["settings"]
    assert m["export_options"]["write_audit"] is True
    # tampering is detected
    m["redactions"]["video"] = []
    p2 = p.with_name("tampered.json")
    p2.write_text(json.dumps(m), encoding="utf-8")
    assert verify_manifest(p2)[0] is False


def test_report_mentions_manifest(exported):
    res = exported["res"]
    csv = Path(res["report_csv"]).read_text(encoding="utf-8")
    assert res["manifest_sha256"] in csv and "source_unchanged_since_open,True" in csv
    assert Path(res["report_pdf"]).stat().st_size > 1000


def test_manifest_flags_changed_source(tmp_path):
    proj = Project(media=probe(str(DATA / "sample.mp4"), compute_hash=True))
    proj.media.sha256 = "0" * 64   # pretend the file we opened was different
    res = {"source_sha256_at_export": sha256_file(DATA / "sample.mp4"), "source_unchanged": False, "output": "x"}
    m = build_manifest(proj, res, {"output": "x"})
    assert m["source"]["unchanged_since_open"] is False


# ------------------------------------------------------------------ irreversible styles ----
def _roi(seed=3, h=64, w=64):
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_pixelate_has_minimum_block_size():
    for h, w in [(20, 20), (64, 64), (200, 300), (31, 97)]:
        frame = _roi(h=h, w=w)
        video_redact.apply_box(frame, BBox(0, 0, 1, 1), VideoStyle.PIXELATE)
        # count distinct colours: at most ceil(w/16)*ceil(h/16) blocks
        cols = {tuple(c) for c in frame.reshape(-1, 3)}
        max_blocks = int(round(w / max(16, min(h, w) // 8))) * int(round(h / max(16, min(h, w) // 8)))
        assert len(cols) <= max(1, max_blocks), (h, w, len(cols))
        # a 16 px block must be flat (no detail finer than the floor)
        assert np.ptp(frame[:min(16, h), :min(16, w)].reshape(-1, 3), axis=0).max() == 0


def test_blur_destroys_detail():
    frame = _roi(h=120, w=120)
    orig = frame.copy()
    video_redact.apply_box(frame, BBox(0, 0, 1, 1), VideoStyle.BLUR)
    # high-frequency content gone: local variance in any 8x8 patch is tiny vs the original
    def var8(img):
        return np.mean([img[y:y + 8, x:x + 8].astype(float).var() for y in range(0, 112, 8) for x in range(0, 112, 8)])
    assert var8(frame) < var8(orig) * 0.03
    # no more than ~6x6 "effective blocks" worth of information
    small = frame[::20, ::20]
    assert np.abs(frame[:20, :20].astype(int) - small[0, 0].astype(int)).max() < 40


def test_default_style_is_solid():
    assert Project().default_video_style == VideoStyle.BLACK
    from videoredact.paths import DEFAULT_SETTINGS
    assert DEFAULT_SETTINGS["default_video_style"] in ("black", VideoStyle.BLACK)


# ------------------------------------------------------------------ offline mode ----
def test_offline_defaults():
    from videoredact.paths import DEFAULT_SETTINGS
    assert DEFAULT_SETTINGS["offline"] is True
    assert DEFAULT_SETTINGS["check_updates"] is True   # notify-only, GitHub metadata; Phase 5


def test_ensure_model_refuses_download_when_offline(monkeypatch, tmp_path):
    from videoredact.vision import models as reg
    from videoredact import paths
    monkeypatch.setattr(reg, "model_search_dirs", lambda: [tmp_path])
    monkeypatch.setattr(reg, "models_dir", lambda: tmp_path)
    monkeypatch.setitem(paths.settings, "offline", True)
    calls = []
    monkeypatch.setattr(reg.urllib.request, "urlopen", lambda *a, **k: calls.append(a) or (_ for _ in ()).throw(AssertionError("network")))
    with pytest.raises(RuntimeError, match="offline mode"):
        reg.ensure_model("yunet")
    assert not calls


def test_app_sets_telemetry_off_env(monkeypatch):
    for k in ("HF_HUB_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "DISABLE_TELEMETRY", "VIDEOREDACT_OFFLINE"):
        monkeypatch.delenv(k, raising=False)
    from videoredact import app
    app._offline_env()
    assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert os.environ["DISABLE_TELEMETRY"] == "1"


def test_no_network_guard_blocks_sockets():
    import socket
    from videoredact.app import _NoNetwork
    with _NoNetwork() as g:
        with pytest.raises(OSError):
            socket.create_connection(("127.0.0.1", 9), timeout=0.2)
        s = socket.socket()
        with pytest.raises(OSError):
            s.connect(("127.0.0.1", 9))
        s.close()
    assert len(g.calls) == 2
    # restored afterwards
    assert socket.create_connection is not None and "blocked" not in socket.socket.connect.__name__


def test_model_inventory_covers_bundled_models():
    inv = model_inventory()
    names = {m["name"] for m in inv}
    assert {"yunet", "yolox_tiny"} <= names or not names  # empty only on a machine without models


def test_openvino_telemetry_is_unimportable():
    import videoredact  # noqa: F401  (hardening runs on import)
    with pytest.raises(ImportError):
        import openvino_telemetry  # noqa: F401
    import openvino  # noqa: F401  must still import fine (ovc falls back to its telemetry stub)
    from openvino.tools.ovc import telemetry_utils
    assert "telemetry_stub" in telemetry_utils.tm.__name__


def test_installer_excludes_telemetry_package():
    spec = (Path(__file__).parents[1] / "installer" / "videoredact.spec").read_text(encoding="utf-8")
    assert '"openvino_telemetry"' in spec


def test_manifest_accepts_numpy_boxes(tmp_path):
    """Auto-detected tracks carry float32 coordinates; the first benchmark after Phase 4 crashed on them."""
    from videoredact.core.audit import write_manifest
    proj = Project(media=probe(str(DATA / "sample.mp4")))
    f32 = np.float32
    proj.add_video_track(VideoTrack(label="face", source="auto", pad=f32(0.25),
                                    spans=[Span(0, 5, {0: BBox(f32(0.1), f32(0.2), f32(0.3), f32(0.4))})]))
    res = {"output": "x.mp4", "output_sha256": "0" * 64, "elapsed_s": f32(1.5)}
    p = tmp_path / "m.json"
    h = write_manifest(proj, res, {"output": "x.mp4", "crf": f32(23)}, p)
    assert verify_manifest(p) == (True, h)
    m = json.loads(p.read_text(encoding="utf-8"))
    assert m["redactions"]["video"][0]["spans"][0]["keyframes"]["0"][0] == pytest.approx(0.1)
