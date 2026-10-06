"""Object detectors running on CPU through OpenCV / ONNX Runtime.

- FaceDetector: YuNet via cv2.FaceDetectorYN (MIT, ~10 ms at 640 px).
- ObjectDetector: YOLOX (Apache-2.0) ONNX for COCO classes, mapped to the
  privacy categories we care about (screen, document, person, phone...).
- CombinedDetector: runs the enabled detectors and returns unified results.

All boxes returned are normalized BBox instances plus a label and score.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from videoredact.core.model import BBox
from . import models as model_registry

# COCO class index -> our privacy label
COCO_TO_LABEL = {
    0: "person",
    62: "screen",   # tv
    63: "screen",   # laptop
    67: "phone",    # cell phone
    73: "document", # book
    66: "keyboard",
    64: "mouse",
}
# Labels the UI can offer for automatic detection
AUTO_LABELS = ["face", "screen", "document", "phone", "person"]
LABEL_DESCRIPTIONS = {
    "face": "Faces (YuNet)",
    "screen": "Computer / TV screens and laptops",
    "document": "Papers, notepads, books",
    "phone": "Cell phones",
    "person": "Whole bodies",
}


@dataclass
class Detection:
    bbox: BBox
    label: str
    score: float


def _nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    if len(boxes_xyxy) == 0:
        return []
    b = [[float(x0), float(y0), float(x1 - x0), float(y1 - y0)] for x0, y0, x1, y1 in boxes_xyxy]
    idx = cv2.dnn.NMSBoxes(b, scores.astype(float).tolist(), 0.0, iou_thr)
    return [int(i) for i in np.array(idx).reshape(-1)]


class FaceDetector:
    def __init__(self, score_thr: float = 0.6, nms_thr: float = 0.3, max_side: int = 960):
        path = model_registry.ensure_model("yunet")
        self.det = cv2.FaceDetectorYN.create(str(path), "", (320, 320), score_thr, nms_thr, 5000)
        self.max_side = max_side
        self._size = (0, 0)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        H, W = frame.shape[:2]
        scale = min(1.0, self.max_side / max(H, W))
        img = cv2.resize(frame, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else frame
        h, w = img.shape[:2]
        if (w, h) != self._size:
            self.det.setInputSize((w, h))
            self._size = (w, h)
        _, faces = self.det.detect(img)
        out: list[Detection] = []
        if faces is None:
            return out
        for f in faces:
            x, y, bw, bh, score = float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[-1])
            out.append(Detection(BBox.from_pixels(x, y, x + bw, y + bh, w, h), "face", score))
        return out


class ObjectDetector:
    """YOLOX ONNX inference (yolox_s @640 or yolox_tiny @416)."""

    def __init__(self, model: str = "yolox_s", conf_thr: float = 0.35, nms_thr: float = 0.45,
                 labels: Optional[set[str]] = None, threads: int = 0):
        import onnxruntime as ort
        path = model_registry.ensure_model(model)
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        shp = inp.shape
        self.size = int(shp[2]) if isinstance(shp[2], int) else (416 if "tiny" in model else 640)
        self.conf_thr = conf_thr
        self.nms_thr = nms_thr
        self.labels = labels  # None = all mapped labels
        self._grids, self._strides = self._make_grids(self.size)

    @staticmethod
    def _make_grids(size: int):
        grids, strides = [], []
        for s in (8, 16, 32):
            n = size // s
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            g = np.stack((xv, yv), 2).reshape(-1, 2)
            grids.append(g)
            strides.append(np.full((g.shape[0], 1), s))
        return np.concatenate(grids, 0).astype(np.float32), np.concatenate(strides, 0).astype(np.float32)

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        H, W = frame.shape[:2]
        r = min(self.size / H, self.size / W)
        nw, nh = int(W * r), int(H * r)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        padded = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        padded[:nh, :nw] = resized
        blob = padded.transpose(2, 0, 1)[None].astype(np.float32)
        return np.ascontiguousarray(blob), r

    def detect(self, frame: np.ndarray) -> list[Detection]:
        H, W = frame.shape[:2]
        blob, r = self._preprocess(frame)
        out = self.sess.run(None, {self.input_name: blob})[0][0]  # [N, 85]
        xy = (out[:, :2] + self._grids) * self._strides
        wh = np.exp(out[:, 2:4]) * self._strides
        obj = out[:, 4:5]
        cls = out[:, 5:]
        scores_all = obj * cls
        cls_id = scores_all.argmax(1)
        scores = scores_all[np.arange(len(cls_id)), cls_id]
        keep = scores > self.conf_thr
        if not keep.any():
            return []
        xy, wh, scores, cls_id = xy[keep], wh[keep], scores[keep], cls_id[keep]
        x0y0 = (xy - wh / 2) / r
        x1y1 = (xy + wh / 2) / r
        boxes = np.concatenate([x0y0, x1y1], 1)
        dets: list[Detection] = []
        for c in np.unique(cls_id):
            label = COCO_TO_LABEL.get(int(c))
            if label is None or (self.labels is not None and label not in self.labels):
                continue
            m = cls_id == c
            for i in _nms(boxes[m], scores[m], self.nms_thr):
                x0, y0, x1, y1 = boxes[m][i]
                dets.append(Detection(BBox.from_pixels(x0, y0, x1, y1, W, H), label, float(scores[m][i])))
        return dets


class CombinedDetector:
    def __init__(self, labels: list[str], face_conf: float = 0.6, obj_conf: float = 0.35,
                 obj_model: str = "yolox_s", threads: int = 0):
        self.labels = set(labels)
        self.face = FaceDetector(face_conf) if "face" in self.labels else None
        obj_labels = self.labels - {"face"}
        self.obj = ObjectDetector(obj_model, obj_conf, labels=obj_labels, threads=threads) if obj_labels else None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        dets: list[Detection] = []
        if self.face:
            dets += self.face.detect(frame)
        if self.obj:
            dets += self.obj.detect(frame)
        return dets
