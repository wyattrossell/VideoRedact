"""Adapters that expose the existing model classes through the Detector interface."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .base import Detection, PerFrameDetector


class YuNetDetector(PerFrameDetector):
    labels = {"face"}

    def __init__(self, score_thr: float = 0.5, max_side: int = 1280):
        from ..detector import FaceDetector
        self.inner = FaceDetector(score_thr, max_side=max_side)
        self.name = f"yunet{max_side}"

    def _detect_frame(self, frame: np.ndarray) -> list[Detection]:
        return [Detection(d.bbox, d.label, d.score, self.name) for d in self.inner.detect(frame)]


class CenterFaceDetector(PerFrameDetector):
    labels = {"face"}

    def __init__(self, score_thr: float = 0.35, max_side: int = 1280, threads: int = 0):
        from ..detector import CenterFaceDetector as _CF
        self.inner = _CF(score_thr, max_side=max_side, threads=threads)
        self.name = f"centerface{max_side}"

    def _detect_frame(self, frame: np.ndarray) -> list[Detection]:
        return [Detection(d.bbox, d.label, d.score, self.name) for d in self.inner.detect(frame)]


class YOLOXDetector(PerFrameDetector):
    def __init__(self, model: str = "yolox_tiny", conf_thr: float = 0.35, labels: Optional[set[str]] = None, threads: int = 0):
        from ..detector import ObjectDetector, COCO_TO_LABEL
        self.inner = ObjectDetector(model, conf_thr, labels=labels, threads=threads)
        self.labels = set(labels) if labels else set(COCO_TO_LABEL.values())
        self.name = model

    def _detect_frame(self, frame: np.ndarray) -> list[Detection]:
        return [Detection(d.bbox, d.label, d.score, self.name) for d in self.inner.detect(frame)]


class EgoBlurDetector(PerFrameDetector):
    """Meta EgoBlur face / license-plate detectors (TorchScript, Apache-2.0 with a
    click-through agreement). Requires torch (evaluation venv) and the .jit files
    downloaded manually from https://www.projectaria.com/tools/egoblur/ into
    <models>/eval/ego_blur_face.jit and ego_blur_lp.jit."""

    def __init__(self, kind: str = "face", score_thr: float = 0.5, model_path: Optional[str] = None, max_side: int = 1280):
        import torch
        from pathlib import Path
        from videoredact.paths import models_dir
        path = Path(model_path) if model_path else models_dir() / "eval" / ("ego_blur_face.jit" if kind == "face" else "ego_blur_lp.jit")
        if not path.exists():
            raise FileNotFoundError(f"EgoBlur model not found: {path} (download from projectaria.com)")
        self.torch = torch
        self.model = torch.jit.load(str(path), map_location="cpu").eval()
        self.label = "face" if kind == "face" else "plate"
        self.labels = {self.label}
        self.score_thr, self.max_side = score_thr, max_side
        self.name = f"egoblur_{kind}"

    def _detect_frame(self, frame: np.ndarray) -> list[Detection]:
        import cv2
        from videoredact.core.model import BBox
        H, W = frame.shape[:2]
        s = min(1.0, self.max_side / max(H, W))
        img = cv2.resize(frame, (int(W * s), int(H * s))) if s < 1 else frame
        t = self.torch.from_numpy(img).permute(2, 0, 1).float()   # EgoBlur takes BGR uint8-range CHW
        with self.torch.no_grad():
            boxes, _, scores, _ = self.model(t)
        out = []
        for b, sc in zip(boxes.numpy(), scores.numpy()):
            if sc < self.score_thr:
                continue
            x1, y1, x2, y2 = b / s
            out.append(Detection(BBox.from_pixels(x1, y1, x2, y2, W, H), self.label, float(sc), self.name))
        return out
