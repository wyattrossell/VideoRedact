"""Licence-plate detectors.

YoloV9PlateDetector (shipped, 0.1.19): `yolo-v9-t-384-license-plates-end2end.onnx` from
open-image-models (ankandrew, MIT; accepted as MIT on 2026-11-02, the way OpenScrub and
PrivacyCam ship it; provenance notes in bench/results/PLATES.md). Letterboxed 384 px RGB input
in 0..1, NMS inside the graph, rows of [batch, x1, y1, x2, y2, class, score] in input pixels.
About 20 ms per 720p frame on the CPU.

On body-cam footage the model fires on anything rectangular with text: the in-car laptop
screen and keyboard, shirts, notepads, a driver's licence, the camera's own timestamp overlay
(contact sheets of 2026-10-29 and 2026-11-02). Two filters make it usable under the
precision policy:

  * the post-filters PrivacyCam applies (aspect ratio 1.1-8, tiny and huge boxes dropped,
    at most 20 per frame), and
  * `VehicleGate`: a plate is kept only when it lies inside a vehicle box (COCO car, truck,
    bus, motorcycle from the YOLOX detector that runs anyway). A plate held in the hand, or on
    a vehicle the object detector misses, needs a drawn box; that is the precision trade.

RTDETRPlateDetector is the 2026-10-29 candidate (Topurrra, Apache-2.0); 850 ms/frame on CPU,
kept for comparison only and not in the model registry.
"""
from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from videoredact.core.model import BBox
from .base import Detection, Detector, PerFrameDetector, _contains

INPUT = 640              # RT-DETR
PLATE_INPUT = 384        # YOLOv9-t
PLATE_LABEL = "license_plate"
VEHICLE_LABEL = "vehicle"
PLATE_MIN_ASPECT = 1.1   # width / height: plates are wider than tall (PrivacyCam's filter)
PLATE_MAX_ASPECT = 8.0
PLATE_MAX_AREA_FRAC = 0.35   # of the model input: a "plate" filling a third of the picture is a screen or a sign
PLATE_MAX_PER_FRAME = 20
PLATE_DEDUPE_IOU = 0.55
PLATE_PAD_Y = 0.10       # extra height each side so the plate's frame and bolts are covered (PrivacyCam: 20 % total)


def decode_plate_rows(rows: np.ndarray, W: int, H: int, r: float, dx: float, dy: float,
                      score_thr: float, source: str = "plate_yolov9t") -> list[Detection]:
    """Rows [batch, x1, y1, x2, y2, class, score] in letterboxed input pixels (scale `r`,
    offsets `dx`, `dy`) -> plate detections normalised to the WxH frame, after the shape
    filters above. Pure, so the filters are testable without the model."""
    out: list[tuple[float, BBox]] = []
    for row in np.asarray(rows, dtype=np.float32).reshape(-1, 7):
        _, x1, y1, x2, y2, _, s = (float(v) for v in row[:7])
        if not all(np.isfinite(v) for v in (x1, y1, x2, y2, s)):
            continue
        if s < score_thr or s > 1.001 or x2 <= x1 or y2 <= y1:
            continue
        mw, mh = x2 - x1, y2 - y1
        if mw < 6 or mh < 3 or not (PLATE_MIN_ASPECT <= mw / mh <= PLATE_MAX_ASPECT):
            continue
        if mw * mh > PLATE_MAX_AREA_FRAC * PLATE_INPUT * PLATE_INPUT:
            continue
        fx0, fy0 = (x1 - dx) / r, (y1 - dy) / r
        fx1, fy1 = (x2 - dx) / r, (y2 - dy) / r
        fx0, fx1 = max(0.0, fx0), min(float(W), fx1)
        fy0, fy1 = max(0.0, fy0), min(float(H), fy1)
        if fx1 - fx0 < 4 or fy1 - fy0 < 3:
            continue
        pad = (fy1 - fy0) * PLATE_PAD_Y
        box = BBox.from_pixels(fx0, max(0.0, fy0 - pad), fx1, min(float(H), fy1 + pad), W, H)
        out.append((s, box))
    out.sort(key=lambda t: -t[0])
    kept: list[Detection] = []
    for s, box in out:
        if any(box.iou(k.bbox) > PLATE_DEDUPE_IOU for k in kept):
            continue
        kept.append(Detection(box, PLATE_LABEL, s, source))
        if len(kept) >= PLATE_MAX_PER_FRAME:
            break
    return kept


def letterbox(frame: np.ndarray, size: int) -> tuple[np.ndarray, float, float, float]:
    """Centre the frame on a size x size grey (114) canvas; returns (canvas, scale, dx, dy)."""
    H, W = frame.shape[:2]
    r = min(size / W, size / H)
    nw, nh = max(1, int(round(W * r))), max(1, int(round(H * r)))
    canvas = np.full((size, size, 3), 114, np.uint8)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, r, float(dx), float(dy)


class YoloV9PlateDetector(PerFrameDetector):
    labels = {PLATE_LABEL}

    def __init__(self, score_thr: float = 0.5, threads: int = 0, prefer_openvino: bool = False):
        from ..runner import make_runner
        from .. import models as reg
        # CPU ONNX Runtime by default: 20 ms/frame, and the end-to-end graph has a dynamic
        # output row count that the OpenVINO shape cache gains nothing from.
        self.runner = make_runner(reg.ensure_model("plate_yolov9t"), threads, prefer_openvino=prefer_openvino)
        self.engine = self.runner.engine
        self.score_thr = float(score_thr)
        self.name = "plate_yolov9t"

    def set_score_threshold(self, thr: float) -> None:
        self.score_thr = float(thr)

    def _detect_frame(self, frame: np.ndarray) -> list[Detection]:
        H, W = frame.shape[:2]
        canvas, r, dx, dy = letterbox(frame, PLATE_INPUT)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        blob = np.ascontiguousarray((rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])
        out = self.runner.run(blob)[0]
        if out is None or out.size == 0:
            return []
        return decode_plate_rows(out, W, H, r, dx, dy, self.score_thr, self.name)


class VehicleGate(Detector):
    """Keep `license_plate` detections only where at least `min_inside` of the plate lies
    inside a `vehicle` detection from the same pass; vehicle boxes themselves are dropped
    unless `keep_vehicles`. Everything else passes through untouched."""

    def __init__(self, inner: Detector, min_inside: float = 0.8, keep_vehicles: bool = False):
        self.inner = inner
        self.min_inside = float(min_inside)
        self.keep_vehicles = keep_vehicles
        self.labels = set(inner.labels) - ({VEHICLE_LABEL} if not keep_vehicles else set())
        self.name = f"vehiclegate({inner.name})"

    def detect(self, frames: Sequence[np.ndarray]) -> list[list[Detection]]:
        out = []
        for dets in self.inner.detect(frames):
            vehicles = [d.bbox for d in dets if d.label == VEHICLE_LABEL]
            kept = []
            for d in dets:
                if d.label == VEHICLE_LABEL:
                    if self.keep_vehicles:
                        kept.append(d)
                    continue
                if d.label == PLATE_LABEL and not any(_contains(v, d.bbox, self.min_inside) for v in vehicles):
                    continue
                kept.append(d)
            out.append(kept)
        return out

    def close(self) -> None:
        self.inner.close()


class RTDETRPlateDetector(PerFrameDetector):
    """2026-10-29 candidate (Topurrra/rtdetr-license-plate-detection-onnx, Apache-2.0): fixed
    1x3x640x640 RGB input scaled to 0..1 (stretched, not letterboxed), outputs `logits`
    (1, 300, 1) and `pred_boxes` (1, 300, 4) as normalized (cx, cy, w, h). Not registered
    (850 ms/frame on CPU); construct with an explicit model path for comparisons."""
    labels = {PLATE_LABEL}

    def __init__(self, model_path: str, score_thr: float = 0.08, threads: int = 0, prefer_openvino: bool = True):
        from ..runner import make_runner
        self.runner = make_runner(model_path, threads, outputs=["logits", "pred_boxes"], prefer_openvino=prefer_openvino)
        self.engine = self.runner.engine
        self.score_thr = score_thr
        self.name = "plate_rtdetr"

    def set_score_threshold(self, thr: float) -> None:
        self.score_thr = float(thr)

    def _detect_frame(self, frame: np.ndarray) -> list[Detection]:
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
                                 PLATE_LABEL, float(s), self.name))
        return out
