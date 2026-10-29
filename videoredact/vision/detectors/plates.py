"""Licence-plate detector: RT-DETRv2-R18 fine-tuned on Open Images "Vehicle registration plate"
(Topurrra/rtdetr-license-plate-detection-onnx; Apache-2.0 code and weights, CC-BY training
data; chosen in the 2026-10-29 licence survey, bench/results/PLATES.md).

Fixed 1x3x640x640 RGB input scaled to 0..1 (the frame is stretched, not letterboxed, as the
model was trained); outputs `logits` (1, 300, 1) and `pred_boxes` (1, 300, 4) as normalized
(cx, cy, w, h). Scores are compressed (the best box is usually 0.1-0.2), so the threshold is
low and the gates downstream (confident-only auto-detect, the anchored tracker's region /
size checks) do the filtering.
"""
from __future__ import annotations

import cv2
import numpy as np

from videoredact.core.model import BBox
from .base import Detection, PerFrameDetector

INPUT = 640


class RTDETRPlateDetector(PerFrameDetector):
    labels = {"license_plate"}

    def __init__(self, score_thr: float = 0.08, threads: int = 0, prefer_openvino: bool = True):
        from ..runner import make_runner
        from .. import models as reg
        self.runner = make_runner(reg.ensure_model("plate_rtdetr"), threads, outputs=["logits", "pred_boxes"],
                                  prefer_openvino=prefer_openvino)
        self.engine = self.runner.engine
        self.score_thr = score_thr
        self.name = "plate_rtdetr"

    def set_score_threshold(self, thr: float) -> None:
        self.score_thr = float(thr)

    def _detect_frame(self, frame: np.ndarray) -> list[Detection]:
        H, W = frame.shape[:2]
        img = cv2.cvtColor(cv2.resize(frame, (INPUT, INPUT), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        blob = np.ascontiguousarray((img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])
        outs = self.runner.run(blob)
        logits, boxes = None, None
        for o in outs:
            if o.ndim == 3 and o.shape[-1] == 4:
                boxes = o[0]
            elif o.ndim == 3:
                logits = o[0]
        if logits is None or boxes is None:
            return []
        scores = 1.0 / (1.0 + np.exp(-logits.max(axis=-1)))
        out = []
        for s, (cx, cy, w, h) in zip(scores, boxes):
            if s < self.score_thr or w <= 0 or h <= 0:
                continue
            out.append(Detection(BBox(float(cx - w / 2), float(cy - h / 2), float(w), float(h)).clamp(),
                                 "license_plate", float(s), self.name))
        return out
