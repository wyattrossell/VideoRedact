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
  * `VehiclePlates`: a plate is kept only when it lies inside a vehicle box (COCO car, truck,
    bus, motorcycle from the YOLOX detector that runs anyway, taken at a low threshold since a
    wrong vehicle box costs nothing). A plate held in the hand needs a drawn box.

Round two (2026-11-02, "plates still not working great"): at 384 px a plate on a car across
the street is a dozen pixels wide and the model misses it. `VehiclePlates` therefore also
runs the plate model on a crop of every detected vehicle (up to four, largest first), where
the same plate is three to five times bigger, and fuses those hits with the full-frame pass.
Plates the gate rejects are passed on marked `plate_ungated`; auto-detect uses them only to
continue a plate track it has already confirmed (the car drifts out of the object detector's
view before the plate does) and never to start one.

RTDETRPlateDetector is the 2026-10-29 candidate (Topurrra, Apache-2.0); 850 ms/frame on CPU,
kept for comparison only and not in the model registry.
"""
from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from videoredact.core.model import BBox
from .base import Detection, Detector, PerFrameDetector, _contains, nms_merge

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
UNGATED_SOURCE = "plate_ungated"   # a plate outside every vehicle box: continuation only (see auto_detect)
VEHICLE_PAD = 0.08       # vehicle boxes are grown by this fraction before gating / cropping (plates hang on the bumper edge)
CROP_MAX_VEHICLES = 4    # vehicle crops searched per frame, largest first
CROP_MIN_SIDE = 48       # px: smaller vehicle boxes hold no readable plate
CROP_MAX_AREA = 0.6      # a vehicle filling this much of the frame gains nothing from a crop (same scale as the full pass)


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

    def __init__(self, score_thr: float = 0.4, threads: int = 0, prefer_openvino: bool = False, enhance: str = "off"):
        from ..runner import make_runner
        from .. import models as reg
        # CPU ONNX Runtime by default: 20 ms/frame, and the end-to-end graph has a dynamic
        # output row count that the OpenVINO shape cache gains nothing from.
        self.runner = make_runner(reg.ensure_model("plate_yolov9t"), threads, prefer_openvino=prefer_openvino)
        self.engine = self.runner.engine
        self.score_thr = float(score_thr)
        from ..enhance import as_mode
        self.enhance = as_mode(enhance)  # off | auto | always: boosted copy of the frame / vehicle crop (vision/enhance.py)
        self.name = "plate_yolov9t"

    def set_score_threshold(self, thr: float) -> None:
        self.score_thr = float(thr)

    def _detect_frame(self, frame: np.ndarray) -> list[Detection]:
        H, W = frame.shape[:2]
        if self.enhance != "off":
            from ..enhance import boost
            frame = boost(frame, self.enhance)
        canvas, r, dx, dy = letterbox(frame, PLATE_INPUT)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        blob = np.ascontiguousarray((rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])
        out = self.runner.run(blob)[0]
        if out is None or out.size == 0:
            return []
        return decode_plate_rows(out, W, H, r, dx, dy, self.score_thr, self.name)


def _grow(b: BBox, frac: float) -> BBox:
    return BBox(b.x - b.w * frac, b.y - b.h * frac, b.w * (1 + 2 * frac), b.h * (1 + 2 * frac)).clamp()


class VehiclePlates(Detector):
    """Plates on vehicles. Wraps the detector that emits `vehicle` boxes (YOLOX) and plates
    (full-frame pass) and:

      * runs `plate_det` again on a crop of each vehicle box (`CROP_MAX_VEHICLES` largest, grown
        by `VEHICLE_PAD`), where a distant plate is several times bigger than in the 384 px
        full-frame pass, and fuses the two sets (`nms_merge`);
      * keeps a plate only where at least `min_inside` of it lies inside a (grown) vehicle box;
        the rest are passed on with source `UNGATED_SOURCE` when `keep_ungated` (auto-detect
        continues an already confirmed plate track with them, never starts one) or dropped;
      * drops the vehicle boxes themselves unless `keep_vehicles`.
    Everything else passes through untouched."""

    def __init__(self, inner: Detector, plate_det: Optional[PerFrameDetector] = None, min_inside: float = 0.6,
                 keep_vehicles: bool = False, keep_ungated: bool = True, crops: bool = True):
        self.inner = inner
        self.plate_det = plate_det
        self.min_inside = float(min_inside)
        self.keep_vehicles = keep_vehicles
        self.keep_ungated = keep_ungated
        self.crops = crops and plate_det is not None
        self.labels = set(inner.labels) - ({VEHICLE_LABEL} if not keep_vehicles else set())
        self.name = f"vehicleplates({inner.name})"

    def _crop_plates(self, frame: np.ndarray, vehicles: list[BBox]) -> list[Detection]:
        H, W = frame.shape[:2]
        out: list[Detection] = []
        for v in sorted(vehicles, key=lambda b: -b.w * b.h)[:CROP_MAX_VEHICLES]:
            if v.w * v.h >= CROP_MAX_AREA:
                continue
            x0, y0, x1, y1 = v.to_pixels(W, H)
            if x1 - x0 < CROP_MIN_SIDE or y1 - y0 < CROP_MIN_SIDE:
                continue
            crop = frame[y0:y1, x0:x1]
            cw, ch = (x1 - x0) / W, (y1 - y0) / H
            for d in self.plate_det._detect_frame(crop):
                b = d.bbox
                out.append(Detection(BBox(x0 / W + b.x * cw, y0 / H + b.y * ch, b.w * cw, b.h * ch).clamp(),
                                     d.label, d.score, d.source + "+crop", getattr(d, "landmarks", None)))
        return out

    def detect(self, frames: Sequence[np.ndarray]) -> list[list[Detection]]:
        out = []
        for frame, dets in zip(frames, self.inner.detect(frames)):
            vehicles = [_grow(d.bbox, VEHICLE_PAD) for d in dets if d.label == VEHICLE_LABEL]
            plates = [d for d in dets if d.label == PLATE_LABEL]
            kept = [d for d in dets if d.label == VEHICLE_LABEL and self.keep_vehicles]
            if self.crops and vehicles:
                plates = nms_merge(plates + self._crop_plates(frame, vehicles), PLATE_DEDUPE_IOU)
            for d in plates:
                if any(_contains(v, d.bbox, self.min_inside) for v in vehicles):
                    kept.append(d)
                elif self.keep_ungated:
                    kept.append(Detection(d.bbox, d.label, d.score, UNGATED_SOURCE, getattr(d, "landmarks", None)))
            kept += [d for d in dets if d.label not in (VEHICLE_LABEL, PLATE_LABEL)]
            out.append(kept)
        return out

    def close(self) -> None:
        self.inner.close()


# 0.1.19 name, kept for callers that import it: the gate without the crop pass or continuation.
def VehicleGate(inner: Detector, min_inside: float = 0.8, keep_vehicles: bool = False) -> VehiclePlates:
    return VehiclePlates(inner, None, min_inside=min_inside, keep_vehicles=keep_vehicles, keep_ungated=False, crops=False)


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
