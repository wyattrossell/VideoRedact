"""Per-export audit manifest (JSON sidecar) for evidence integrity.

Written next to every exported file as `<output>_audit.json`. Contents:
  operator, timestamps, software version, machine, inference tier,
  SHA-256 of the source (at open and re-checked at export), of the output,
  of the project file and of every model file used (with licence),
  the settings/thresholds in force, the complete list of redactions
  (audio ranges; video tracks with every keyframe box and frame range),
  the project's audit log (every manual edit), the export options and the
  verification result. A SHA-256 of the manifest body is stored inside the
  manifest and printed in the PDF report so the two can be cross-checked.
"""
from __future__ import annotations

import datetime as _dt
import getpass
import hashlib
import json
import os
import platform
import socket
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from videoredact import __version__
from .media import sha256_file
from .model import Project, as_enum, AudioStyle, VideoStyle, Shape

_MODEL_HASH_CACHE: dict[str, str] = {}


def model_inventory() -> list[dict[str, Any]]:
    """Every model file the app can use: name, file, size, SHA-256, licence, purpose."""
    from videoredact.vision import models as reg
    from videoredact.paths import model_search_dirs
    out = []
    for name, spec in reg.MODELS.items():
        for d in model_search_dirs():
            p = d / spec.filename
            if p.exists():
                key = f"{p}|{p.stat().st_size}|{int(p.stat().st_mtime)}"
                h = _MODEL_HASH_CACHE.get(key)
                if h is None:
                    h = sha256_file(p)
                    _MODEL_HASH_CACHE[key] = h
                out.append({"name": name, "file": str(p), "bytes": p.stat().st_size, "sha256": h,
                            "license": spec.license, "purpose": spec.purpose, "source_url": spec.url})
                break
    # speech model (faster-whisper CTranslate2 folder): hash the model.bin
    try:
        from videoredact.paths import settings, whisper_dir
        size = settings.get("whisper_model", "small")
        root = whisper_dir(size) / f"models--Systran--faster-whisper-{size}" / "snapshots"
        if root.exists():
            for snap in root.iterdir():
                mb = snap / "model.bin"
                if mb.exists():
                    key = f"{mb}|{mb.stat().st_size}"
                    h = _MODEL_HASH_CACHE.get(key) or sha256_file(mb)
                    _MODEL_HASH_CACHE[key] = h
                    out.append({"name": f"faster-whisper-{size}", "file": str(mb), "bytes": mb.stat().st_size, "sha256": h,
                                "license": "MIT (Systran/OpenAI Whisper)", "purpose": "speech-to-text"})
                    break
    except Exception:
        pass
    return out


def engine_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        from videoredact.vision.accel import detect_hardware
        hw = detect_hardware()
        info["tier"] = hw.tier
        info["provider"] = hw.active_provider
        info["hardware"] = hw.summary()
    except Exception as e:  # noqa: BLE001
        info["error"] = str(e)
    try:
        import onnxruntime
        info["onnxruntime"] = onnxruntime.__version__
    except Exception:
        pass
    try:
        import openvino
        info["openvino"] = openvino.__version__
    except Exception:
        pass
    try:
        import cv2
        info["opencv"] = cv2.__version__
    except Exception:
        pass
    return info


def settings_snapshot() -> dict[str, Any]:
    from videoredact.paths import settings
    keys = ["whisper_model", "language", "vad", "default_audio_style", "default_video_style", "beep_frequency",
            "audio_pad_s", "detect_stride", "detect_conf", "face_conf", "detect_classes", "video_pad", "face_pad",
            "detector_model", "face_detector", "tracker", "track_step", "assoc", "lead_frames", "detect_carry",
            "detect_tiles", "detect_multiscale", "detect_head", "export_crf", "export_preset", "verify_export",
            "offline", "check_updates"]
    return {k: settings.get(k) for k in keys if k in settings}


def _redactions(project: Project) -> dict[str, Any]:
    fps = project.media.fps or 30.0
    audio = [{"id": r.id, "start_s": round(r.start, 3), "end_s": round(r.end, 3), "text": r.text, "reason": r.reason,
              "style": (as_enum(AudioStyle, r.style) or as_enum(AudioStyle, project.default_audio_style)).value,
              "source": r.source, "enabled": r.enabled, "created_by": r.created_by, "created_at": r.created_at}
             for r in project.audio_redactions]
    video = []
    for t in project.video_tracks:
        spans = []
        for s in t.spans:
            spans.append({"start_frame": s.start_frame, "end_frame": s.end_frame,
                          "start_s": round(s.start_frame / fps, 3), "end_s": round((s.end_frame + 1) / fps, 3),
                          "keyframes": {str(k): v.to_list() for k, v in sorted(s.keyframes.items())}})
        video.append({"id": t.id, "label": t.label, "style": (as_enum(VideoStyle, t.style) or as_enum(VideoStyle, project.default_video_style)).value,
                      "shape": as_enum(Shape, t.shape).value, "pad": t.pad, "source": t.source, "enabled": t.enabled,
                      "note": t.note, "created_by": t.created_by, "created_at": t.created_at, "spans": spans})
    return {"audio": audio, "video": video}


def build_manifest(project: Project, result: dict[str, Any], export_opts: Any) -> dict[str, Any]:
    now = _dt.datetime.now().astimezone()
    vr = result.get("verify")
    manifest: dict[str, Any] = {
        "format": "videoredact-audit",
        "format_version": 1,
        "software": {"name": "VideoRedact", "version": __version__, "license": "MIT",
                     "repository": "https://github.com/wyattrossell/VideoRedact"},
        "generated_at": now.isoformat(timespec="seconds"),
        "operator": {"name": project.author or "", "windows_user": getpass.getuser(), "machine": socket.gethostname(),
                     "os": f"{platform.system()} {platform.release()} ({platform.version()})"},
        "case_number": project.case_number,
        "notes": project.notes,
        "source": {"path": project.media.path, "sha256_at_open": project.media.sha256,
                   "sha256_at_export": result.get("source_sha256_at_export"),
                   "unchanged_since_open": result.get("source_unchanged"),
                   "duration_s": project.media.duration, "width": project.media.width, "height": project.media.height,
                   "fps": project.media.fps, "frame_count": project.media.frame_count,
                   "video_codec": project.media.video_codec, "audio_codec": project.media.audio_codec},
        "output": {"path": result.get("output"), "sha256": result.get("output_sha256"),
                   "clip_range_s": list(result["clip"]) if result.get("clip") else None,
                   "project_file": result.get("project"), "project_sha256": sha256_file(result["project"]) if result.get("project") and Path(result["project"]).exists() else None,
                   "report_pdf": result.get("report_pdf"), "report_csv": result.get("report_csv"),
                   "verification_csv": result.get("verification_csv"), "elapsed_s": round(result.get("elapsed_s", 0.0), 1)},
        "export_options": {k: v for k, v in (asdict(export_opts) if hasattr(export_opts, "__dataclass_fields__") else dict(export_opts)).items()},
        "engine": engine_info(),
        "models": model_inventory(),
        "transcription_model": project.transcript_model,
        "settings": settings_snapshot(),
        "redactions": _redactions(project),
        "verification": None if vr is None else {"frames_checked": vr.frames_checked, "detections": vr.detections,
                                                 "labels": vr.labels, "uncovered": [
                                                     {"frame": u.frame, "time_s": round(u.time, 3), "label": u.label,
                                                      "score": round(u.score, 3), "covered": round(u.covered, 3), "box": u.bbox.to_list()}
                                                     for u in vr.uncovered]},
        "audit_log": [asdict(e) for e in project.audit_log],
    }
    body = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    manifest["manifest_sha256"] = hashlib.sha256(body).hexdigest()
    return manifest


def write_manifest(project: Project, result: dict[str, Any], export_opts: Any, path: str | Path) -> str:
    m = build_manifest(project, result, export_opts)
    Path(path).write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding="utf-8")
    return m["manifest_sha256"]


def verify_manifest(path: str | Path) -> tuple[bool, str]:
    """Recompute the embedded manifest hash. Returns (ok, stored_hash)."""
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    stored = m.pop("manifest_sha256", "")
    body = json.dumps(m, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(body).hexdigest() == stored, stored
