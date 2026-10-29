"""Face identity embeddings (SFace, OpenCV Zoo, Apache-2.0).

Used by the anchored tracker to decide whether a face found after a loss is the same person
as the one the user boxed. SFace needs the five YuNet landmarks for alignment, so only
detections that carry `landmarks` can be embedded. Everything degrades gracefully: when the
model file is absent `FaceIdentity.try_create()` returns None and callers fall back to
grey-template / colour-histogram comparisons.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .detectors.base import Detection

SAME_PERSON = 0.363   # cosine similarity threshold documented for SFace (OpenCV face recognition sample)


class FaceIdentity:
    def __init__(self):
        from . import models as reg
        if not reg.is_available("sface"):
            raise FileNotFoundError("SFace model not installed")
        self.rec = cv2.FaceRecognizerSF.create(str(reg.model_path("sface")), "")

    @classmethod
    def try_create(cls) -> Optional["FaceIdentity"]:
        try:
            return cls()
        except Exception:  # noqa: BLE001  (model missing or OpenCV built without the module)
            return None

    def embed(self, frame: np.ndarray, det: Detection) -> Optional[np.ndarray]:
        """Unit-length 128-d embedding of the face `det` in `frame`, or None without landmarks."""
        if det.landmarks is None:
            return None
        H, W = frame.shape[:2]
        b = det.bbox
        lm = np.asarray(det.landmarks, dtype=np.float32).reshape(5, 2) * np.array([W, H], dtype=np.float32)
        row = np.concatenate([np.array([b.x * W, b.y * H, b.w * W, b.h * H], dtype=np.float32),
                              lm.ravel(), np.array([det.score], dtype=np.float32)])
        try:
            aligned = self.rec.alignCrop(frame, row)
            feat = self.rec.feature(aligned)
        except cv2.error:
            return None
        f = np.asarray(feat, dtype=np.float32).ravel()
        n = float(np.linalg.norm(f))
        return f / n if n > 0 else None

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))
