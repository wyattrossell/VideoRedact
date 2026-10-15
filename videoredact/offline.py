"""Offline hardening, applied the moment the `videoredact` package is imported.

Runs in every process (GUI, --selftest, multiprocessing workers, tests) BEFORE any
third-party library is imported, so no library can reach the network or report usage:

* openvino          -> `import openvino` pulls in openvino.tools.ovc, which initialises the
                       `openvino_telemetry` package and, with no consent file present, treats the
                       user as opted-IN and posts to www.google-analytics.com. We make that
                       package un-importable (sys.modules entry = None), which makes ovc fall back
                       to its built-in no-op telemetry stub. The package is also excluded from the
                       installer (installer/videoredact.spec).
* huggingface_hub   -> HF_HUB_OFFLINE / HF_HUB_DISABLE_TELEMETRY (used by faster-whisper only
                       for model lookup; the speech model is bundled and loaded local_files_only).
* onnxruntime       -> disable_telemetry_events() is called where sessions are created
                       (vision/detector.py); ORT's telemetry is Windows ETW, never network.
* generic opt-outs  -> DISABLE_TELEMETRY, DO_NOT_TRACK.
* VIDEOREDACT_OFFLINE=1 (when settings["offline"] is true) makes vision/models.ensure_model and
  audio/transcribe refuse to download anything.

Verification: `VideoRedact.exe --selftest --offline-check <file>` runs the full self-test with
every socket connect() replaced by a function that raises and records the attempt; exit code 2
and the printed address list mean a library tried to go online. See docs/CJIS_COMPLIANCE.md.
"""
from __future__ import annotations

import os
import sys

BLOCKED_MODULES = ("openvino_telemetry",)

ENV = {
    "HF_HUB_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    "DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    "OPENVINO_TELEMETRY_OPT_IN": "0",
    "YOLO_OFFLINE": "1",   # Ultralytics is not shipped; documented guard in case it is ever added
}


def harden() -> None:
    for k, v in ENV.items():
        os.environ.setdefault(k, v)
    for name in BLOCKED_MODULES:
        if name in sys.modules and sys.modules[name] is not None:
            continue            # already imported by someone else; nothing we can do here
        sys.modules[name] = None  # type: ignore[assignment]  -> ImportError on import


def is_offline() -> bool:
    """True when runtime downloads are forbidden (settings or environment)."""
    if os.environ.get("VIDEOREDACT_OFFLINE") == "1":
        return True
    try:
        from videoredact.paths import settings
        return bool(settings.get("offline", True))
    except Exception:
        return True


harden()
