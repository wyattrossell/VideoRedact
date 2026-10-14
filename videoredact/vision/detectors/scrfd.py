"""SCRFD face detector (InsightFace) through ONNX Runtime.

Decoding follows the reference implementation (insightface/python-package,
MIT code). LICENSE NOTE: the published SCRFD weights (det_10g.onnx from the
buffalo_l pack, det_34g) are released for non-commercial research use only.
This adapter exists to *evaluate* SCRFD against our footage; shipping it in
the installer needs a licence from InsightFace or a self-trained model.
Model file expected at <models>/eval/det_10g.onnx (see bench/README.md).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from videoredact.core.model import BBox
from videoredact.paths import models_dir
from .base import Detection, PerFrameDetector


class SCRFDDetector(PerFrameDetector):
    name = "scrfd"
    labels = {"face"}

    def __init__(self, model_path: Optional[str] = None, score_thr: float = 0.5, nms_thr: float = 0.4,
                 input_size: int = 640, threads: int = 0):
        from ..detector import _ort_session
        path = Path(model_path) if model_path else models_dir() / "eval" / "det_10g.onnx"
        if not path.exists():
            raise FileNotFoundError(f"SCRFD model not found: {path}")
        self.sess = _ort_session(str(path), threads)
        self.inp = self.sess.get_inputs()[0].name
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.score_thr, self.nms_thr, self.size = score_thr, nms_thr, input_size
        self.fmc = 3
        self.strides = [8, 16, 32]
        self.num_anchors = 2
        self._centers: dict = {}
        self.name = f"scrfd{input_size}"

    def _anchor_centers(self, h: int, w: int, stride: int) -> np.ndarray:
        key = (h, w, stride)
        if key not in self._centers:
            xs, ys = np.mgrid[:h, :w][::-1]
            c = np.stack([xs, ys], axis=-1).astype(np.float32).reshape(-1, 2) * stride
            self._centers[key] = np.stack([c] * self.num_anchors, axis=1).reshape(-1, 2)
        return self._centers[key]

    def _detect_frame(self, frame: np.ndarray) -> list[Detection]:
        H, W = frame.shape[:2]
        s = self.size / max(H, W)
        nw, nh = int(W * s), int(H * s)
        img = cv2.resize(frame, (nw, nh))
        canvas = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        canvas[:nh, :nw] = img
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128.0, (self.size, self.size), (127.5, 127.5, 127.5), swapRB=True)
        outs = self.sess.run(self.out_names, {self.inp: blob})
        scores_l, boxes_l = [], []
        for i, stride in enumerate(self.strides):
            scores = outs[i].reshape(-1)
            bbox_preds = outs[i + self.fmc].reshape(-1, 4) * stride
            h, w = self.size // stride, self.size // stride
            centers = self._anchor_centers(h, w, stride)
            keep = np.where(scores >= self.score_thr)[0]
            if len(keep) == 0:
                continue
            c = centers[keep]
            d = bbox_preds[keep]
            x1, y1 = c[:, 0] - d[:, 0], c[:, 1] - d[:, 1]
            x2, y2 = c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]
            boxes_l.append(np.stack([x1, y1, x2, y2], axis=1) / s)
            scores_l.append(scores[keep])
        if not boxes_l:
            return []
        boxes = np.concatenate(boxes_l)
        scores = np.concatenate(scores_l)
        order = scores.argsort()[::-1]
        boxes, scores = boxes[order], scores[order]
        idx = cv2.dnn.NMSBoxes([[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes],
                               scores.astype(float).tolist(), self.score_thr, self.nms_thr)
        out = []
        for i in np.array(idx).reshape(-1):
            x1, y1, x2, y2 = boxes[i]
            out.append(Detection(BBox.from_pixels(x1, y1, x2, y2, W, H), "face", float(scores[i]), self.name))
        return out
