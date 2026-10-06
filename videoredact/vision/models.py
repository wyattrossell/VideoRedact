"""Model registry and downloader.

Every model here is permissively licensed (MIT / Apache-2.0). Files are
stored under videoredact.paths.models_dir(); scripts/fetch_models.py
pre-downloads them for bundling into the installer.
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from videoredact.paths import models_dir

ProgressCB = Callable[[float, str], None]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    filename: str
    url: str
    license: str
    purpose: str
    min_bytes: int  # sanity check so a Git-LFS pointer or HTML page is rejected


MODELS: dict[str, ModelSpec] = {
    "yunet": ModelSpec(
        "yunet", "face_detection_yunet_2023mar.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "MIT", "Face detector (fast, OpenCV FaceDetectorYN)", 200_000),
    "centerface": ModelSpec(
        "centerface", "centerface.onnx",
        "https://github.com/Star-Clouds/CenterFace/raw/master/models/onnx/centerface.onnx",
        "MIT", "Face detector (higher recall on small faces)", 5_000_000),
    "yolox_s": ModelSpec(
        "yolox_s", "yolox_s.onnx",
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx",
        "Apache-2.0", "COCO object detector: screens, laptops, phones, books, people (640px)", 30_000_000),
    "yolox_tiny": ModelSpec(
        "yolox_tiny", "yolox_tiny.onnx",
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx",
        "Apache-2.0", "COCO object detector, faster/less accurate (416px)", 15_000_000),
}


def model_path(name: str) -> Path:
    return models_dir() / MODELS[name].filename


def is_available(name: str) -> bool:
    p = model_path(name)
    return p.exists() and p.stat().st_size >= MODELS[name].min_bytes


def ensure_model(name: str, progress: Optional[ProgressCB] = None) -> Path:
    """Download the model if missing. Raises on failure (e.g. no internet)."""
    spec = MODELS[name]
    dest = model_path(name)
    if is_available(name):
        return dest
    tmp = dest.with_suffix(".part")
    req = urllib.request.Request(spec.url, headers={"User-Agent": "VideoRedact/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if progress and total:
                progress(done / total, f"Downloading {spec.filename} ({done // 1_000_000} / {total // 1_000_000} MB)")
    if tmp.stat().st_size < spec.min_bytes:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded {spec.filename} is too small - URL may have changed: {spec.url}")
    tmp.replace(dest)
    return dest
