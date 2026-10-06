"""Well-known directories and persisted user settings.

Models live outside the repo so they can be large and shared:
  %LOCALAPPDATA%/VideoRedact/models   (default, user-writable)
  <app root>/models                   (bundled by the installer; checked first)
Override with VIDEOREDACT_MODELS.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

APP_NAME = "VideoRedact"


def app_root() -> Path:
    """Directory of the executable (frozen) or the repo root (source)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parents[1]


def resource_root() -> Path:
    """Where bundled read-only resources (bin/, models/) live.
    PyInstaller one-folder builds put data files in _internal (sys._MEIPASS)."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return app_root()


def log_dir() -> Path:
    d = user_data_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def user_data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    d = Path(base) / APP_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_dir() -> Path:
    env = os.environ.get("VIDEOREDACT_MODELS")
    if env:
        p = Path(env)
        p.mkdir(parents=True, exist_ok=True)
        return p
    for bundled in (resource_root() / "models", app_root() / "models"):
        if getattr(sys, "frozen", False) and bundled.exists() and any(bundled.iterdir()):
            return bundled
    d = user_data_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def whisper_dir() -> Path:
    d = models_dir() / "whisper"
    d.mkdir(parents=True, exist_ok=True)
    return d


def settings_path() -> Path:
    return user_data_dir() / "settings.json"


DEFAULT_SETTINGS: dict[str, Any] = {
    "author": os.environ.get("USERNAME") or os.environ.get("USER") or "",
    "whisper_model": "small",
    "language": "",                 # "" = auto-detect
    "default_audio_style": "beep",
    "default_video_style": "black",
    "beep_frequency": 1000.0,
    "audio_pad_s": 0.05,            # extra padding around each audio redaction at export
    "detect_stride": 3,             # run object detectors every N frames
    "detect_conf": 0.35,
    "face_conf": 0.6,
    "detect_classes": ["face", "screen", "document", "license_plate"],
    "video_pad": 0.10,
    "export_crf": 18,
    "export_preset": "veryfast",
    "cpu_threads": 0,               # 0 = auto
    "tracker": "csrt",
    "media_backend": "",            # unused since the custom player; kept for settings compatibility
    "detector_model": "yolox_s",
    "check_updates": True,          # ask GitHub Releases for a newer installer at startup
    "skip_version": "",             # user chose "skip this version"
}


class Settings(dict):
    def __init__(self):
        super().__init__(DEFAULT_SETTINGS)
        try:
            self.update(json.loads(settings_path().read_text(encoding="utf-8")))
        except Exception:
            pass

    def save(self) -> None:
        settings_path().write_text(json.dumps(self, indent=1), encoding="utf-8")


settings = Settings()
