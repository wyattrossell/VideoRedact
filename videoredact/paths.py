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
from typing import Any, Optional

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
    """Writable models directory (downloads land here)."""
    env = os.environ.get("VIDEOREDACT_MODELS")
    if env:
        p = Path(env)
        p.mkdir(parents=True, exist_ok=True)
        return p
    d = user_data_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def bundled_models_dir() -> Optional[Path]:
    """Read-only models shipped with the installer (frozen builds), if any."""
    if not getattr(sys, "frozen", False):
        return None
    for d in (resource_root() / "models", app_root() / "models"):
        if d.exists() and any(d.iterdir()):
            return d
    return None


def model_search_dirs() -> list[Path]:
    dirs = []
    b = bundled_models_dir()
    if b:
        dirs.append(b)
    dirs.append(models_dir())
    return dirs


def whisper_dir(model_size: str = "") -> Path:
    """Folder to hand faster-whisper as download_root. Prefers the bundled
    folder when it already holds the requested model, else the user folder."""
    if model_size:
        b = bundled_models_dir()
        if b and (b / "whisper" / f"models--Systran--faster-whisper-{model_size}").exists():
            return b / "whisper"
    d = models_dir() / "whisper"
    d.mkdir(parents=True, exist_ok=True)
    return d


def settings_path() -> Path:
    return user_data_dir() / "settings.json"


DEFAULT_SETTINGS: dict[str, Any] = {
    "author": os.environ.get("USERNAME") or os.environ.get("USER") or "",
    "whisper_model": "small",
    "language": "en",               # ISO code; "" = auto-detect (on noisy body-cam openings auto-detect picked
                                    # random scripts and ran 4x slower: Samples 2026-10-16, see CHANGELOG 0.1.12)
    "default_audio_style": "beep",
    "default_video_style": "black",
    "beep_frequency": 1000.0,
    "audio_pad_s": 0.05,            # extra padding around each audio redaction at export
    "detect_stride": 5,             # run object detectors every N frames
    "detect_conf": 0.35,
    "face_conf": 0.5,
    "detect_classes": ["face", "screen", "document", "license_plate"],
    "video_pad": 0.10,
    "export_crf": 18,
    "export_preset": "veryfast",
    "cpu_threads": 0,               # 0 = auto
    "tracker": "vit",              # vit | kcf | csrt (kcf/csrt need opencv-contrib)
    "media_backend": "",            # unused since the custom player; kept for settings compatibility
    "detector_model": "yolox_tiny",
    "check_updates": True,          # the only network call the app makes: GitHub Releases metadata, notify + link, no install
    "offline": True,                # no model downloads at runtime; bundled/pre-fetched models only
    "skip_version": "",             # user chose "skip this version"
    "vad": False,                   # whisper voice-activity filter; off: catches quiet speech in noisy body-cam audio
    "track_step": 2,                # tracker processes every N-th frame
    "track_backward": True,
    "face_detector": "yunet",       # yunet | centerface
    "face_pad": 0.25,               # dilation of auto-detected face/head boxes
    "lead_frames": 15,              # back-fill / persistence around detections
    "assoc": "botsort",             # botsort | bytetrack | ocsort | iou
    "detect_head": False,           # head fallback: about 1 in 3 of its boxes was a real head on the reference clip
    "detect_tiles": False,
    "detect_multiscale": True,
    "detect_precise": True,         # only keep confident detections (AutoDetectOptions defaults); False = recall profile
    "track_lost_limit_s": 8.0,      # manual tracking stops once the object has been gone this long
    "track_search_to_end": False,   # manual tracking keeps searching to the end (object leaves and returns)
    "quick_box": True,              # a drawn box is tracked immediately with the last label/shape/style; Shift+drag asks
    "last_box_label": "face",
    "last_box_shape": None,         # None = ellipse for faces, rectangle otherwise
    "last_box_style": None,         # None = project default
}


SETTINGS_REV = 2   # bump with every entry added to Settings._migrate()


class Settings(dict):
    def __init__(self):
        super().__init__(DEFAULT_SETTINGS)
        try:
            self.update(json.loads(settings_path().read_text(encoding="utf-8")))
        except Exception:
            pass
        self._migrate()

    def _migrate(self) -> None:
        """One-time changes to values older versions saved explicitly."""
        rev = int(self.get("settings_rev", 0) or 0)
        if rev < 1:
            # 0.1.x saved language "" (auto-detect) for everyone; the default is now "en"
            if not self.get("language"):
                self["language"] = DEFAULT_SETTINGS["language"]
            self["settings_rev"] = 1
        if rev < 2:
            # 0.1.15: precision rework. Every install saved detect_head=True from the dialog;
            # the head fallback is now off by default (measured ~1/3 real on the reference clip).
            self["detect_head"] = False
            self["detect_precise"] = True
            self["settings_rev"] = 2
        if rev < 2:
            try:
                self.save()
            except Exception:
                pass

    def save(self) -> None:
        settings_path().write_text(json.dumps(self, indent=1), encoding="utf-8")


settings = Settings()
