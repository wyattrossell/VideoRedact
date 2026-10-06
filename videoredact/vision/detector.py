"""Object detectors running on CPU through OpenCV / ONNX Runtime.

- FaceDetector: YuNet via cv2.FaceDetectorYN (MIT, ~10 ms at 640 px).
- ObjectDetector: YOLOX (Apache-2.0) ONNX for COCO classes, mapped to the
  privacy categories we care about (screen, document, person, phone...).
- CombinedDetector: runs the enabled detectors and returns unified results.

All boxes returned are normalized BBox instances plus a label and score.
"""
from __future__ import annotations

import os
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


def _ort_session(model_bytes_or_path, threads: int = 0):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads or max(1, min(4, (os.cpu_count() or 4) // 2))
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_bytes_or_path, so, providers=["CPUExecutionProvider"])


class CenterFaceDetector:
    """CenterFace (MIT) via ONNX Runtime. Better recall than YuNet on small and
    side-view faces, ~3x slower. Preprocessing follows deface: resize to a
    multiple of 32, no normalisation, BGR. The published model has static
    input dims, so a dynamic-shape copy is written next to it on first use."""

    def __init__(self, score_thr: float = 0.35, nms_thr: float = 0.3, max_side: int = 1280, threads: int = 0):
        from videoredact.paths import models_dir
        src = model_registry.ensure_model("centerface")
        dyn = models_dir() / "centerface_dyn.onnx"
        if not dyn.exists() or dyn.stat().st_size < 1_000_000:
            import onnx
            m = onnx.load(str(src))
            init_names = {i.name for i in m.graph.initializer}
            for inp in m.graph.input:
                if inp.name in init_names:
                    continue
                dims = inp.type.tensor_type.shape.dim
                for i, nm in enumerate(["B", "C", "H", "W"]):
                    if i < len(dims) and i != 1:
                        dims[i].dim_param = nm
            for out in m.graph.output:
                dims = out.type.tensor_type.shape.dim
                for i, nm in enumerate(["B", "C", "H4", "W4"]):
                    if i < len(dims) and i != 1:
                        dims[i].dim_param = nm
            onnx.save(m, str(dyn))
        self.sess = _ort_session(str(dyn), threads)
        self.inp = self.sess.get_inputs()[0].name
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        self.max_side = max_side

    def detect(self, frame: np.ndarray) -> list[Detection]:
        H, W = frame.shape[:2]
        s = min(1.0, self.max_side / max(H, W))
        w32, h32 = max(32, (int(W * s) // 32) * 32), max(32, (int(H * s) // 32) * 32)
        res = cv2.resize(frame, (w32, h32), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(res, scalefactor=1.0, size=(w32, h32), mean=(0, 0, 0), swapRB=False, crop=False)
        heat, scale, off, _lms = self.sess.run(None, {self.inp: blob})
        heat, scale, off = heat[0, 0], scale[0], off[0]
        ys, xs = np.where(heat > self.score_thr)
        if len(ys) == 0:
            return []
        sx, sy = W / w32, H / h32
        boxes, scores = [], []
        for y, x in zip(ys, xs):
            bh, bw = float(np.exp(scale[0, y, x]) * 4), float(np.exp(scale[1, y, x]) * 4)
            cx, cy = (x + float(off[1, y, x]) + 0.5) * 4, (y + float(off[0, y, x]) + 0.5) * 4
            boxes.append([(cx - bw / 2) * sx, (cy - bh / 2) * sy, bw * sx, bh * sy])
            scores.append(float(heat[y, x]))
        idx = cv2.dnn.NMSBoxes(boxes, scores, self.score_thr, self.nms_thr)
        out = []
        for i in np.array(idx).reshape(-1):
            x0, y0, bw, bh = boxes[i]
            out.append(Detection(BBox.from_pixels(x0, y0, x0 + bw, y0 + bh, W, H), "face", scores[i]))
        return out


class ObjectDetector:
    """YOLOX ONNX inference (yolox_s @640 or yolox_tiny @416)."""

    def __init__(self, model: str = "yolox_s", conf_thr: float = 0.35, nms_thr: float = 0.45,
                 labels: Optional[set[str]] = None, threads: int = 0):
        import onnxruntime as ort
        path = model_registry.ensure_model(model)
        so = ort.SessionOptions()
        # Leave cores for OpenCV/decoding and stop ORT threads from spin-waiting after a run:
        # with spinning on, a resident session slowed KCF 2.5x and YOLOX itself 5x (measured).
        so.intra_op_num_threads = threads or max(1, min(4, (os.cpu_count() or 4) // 2))
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
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
    def __init__(self, labels: list[str], face_conf: float = 0.5, obj_conf: float = 0.35,
                 obj_model: str = "yolox_tiny", threads: int = 0, face_model: str = "yunet"):
        self.labels = set(labels)
        self.face = None
        if "face" in self.labels:
            if face_model == "centerface":
                self.face = CenterFaceDetector(min(face_conf, 0.5), threads=threads)
            else:
                self.face = FaceDetector(face_conf, max_side=1280)
        obj_labels = self.labels - {"face"}
        self.obj = ObjectDetector(obj_model, obj_conf, labels=obj_labels, threads=threads) if obj_labels else None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        dets: list[Detection] = []
        if self.face:
            dets += self.face.detect(frame)
        if self.obj:
            dets += self.obj.detect(frame)
        return dets
