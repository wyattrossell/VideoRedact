"""YuNet face detector on ONNX Runtime (same opencv_zoo weights, MIT).

Why: through OpenCV's DNN the 1280x736 pass costs 26-29 ms; through ONNX
Runtime with dynamic input dims it costs ~8 ms on the same CPU (measured
2026-10-15), and it inherits the accelerated provider tiers (OpenVINO /
CUDA) from vision/accel.py, which OpenCV DNN cannot use.

Post-processing follows OpenCV's FaceDetectorYN: for strides 8/16/32 the
model outputs cls, obj, bbox (cx, cy, log w, log h relative to the grid cell)
and 5 landmarks; score = sqrt(cls * obj); NMS per frame.
Input: BGR, 0-255 float32, NCHW, H and W multiples of 32 (frame is padded).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from videoredact.core.model import BBox
from videoredact.paths import models_dir
from .base import Detection, PerFrameDetector

STRIDES = (8, 16, 32)


def _dynamic_model_path() -> Path:
    """Write a copy of the YuNet ONNX with dynamic H/W (and anchor) dims next to the
    user models dir on first use, so ORT can run any frame size."""
    from .. import models as reg
    src = reg.ensure_model("yunet")
    dst = models_dir() / "face_detection_yunet_2023mar_dyn.onnx"
    if dst.exists() and dst.stat().st_size > 100_000:
        return dst
    import onnx
    m = onnx.load(str(src))
    init = {i.name for i in m.graph.initializer}
    for t in m.graph.input:
        if t.name in init:
            continue
        d = t.type.tensor_type.shape.dim
        d[2].dim_param, d[3].dim_param = "H", "W"
    for o in m.graph.output:
        d = o.type.tensor_type.shape.dim
        d[1].dim_param = "A_" + o.name
    onnx.save(m, str(dst))
    return dst


class YuNetORTDetector(PerFrameDetector):
    labels = {"face"}

    def __init__(self, score_thr: float = 0.5, nms_thr: float = 0.3, max_side: int = 1280, threads: int = 0,
                 top_k: int = 5000, prefer_openvino: bool = True):
        """prefer_openvino=False keeps the model on the CPU runtime: the Intel GPU path recompiles
        for every new input shape (~450 ms), which is fine for whole frames of one size and
        hopeless for the varying crops the anchored tracker feeds it."""
        from ..runner import make_runner
        self.out_names = [f"{k}_{s}" for s in STRIDES for k in ("cls", "obj", "bbox", "kps")]
        self.runner = make_runner(_dynamic_model_path(), threads, outputs=self.out_names, nhwc_u8=True,
                                  prefer_openvino=prefer_openvino)
        self.engine = self.runner.engine
        self.score_thr, self.nms_thr, self.max_side, self.top_k = score_thr, nms_thr, max_side, top_k
        self.name = f"yunet_ort{max_side}"
        self._grid: dict = {}

    def set_score_threshold(self, thr: float) -> None:
        self.score_thr = float(thr)

    def _grids(self, h: int, w: int):
        key = (h, w)
        if key not in self._grid:
            g = {}
            for s in STRIDES:
                gh, gw = h // s, w // s
                ys, xs = np.mgrid[:gh, :gw]
                g[s] = (xs.reshape(-1).astype(np.float32), ys.reshape(-1).astype(np.float32))
            self._grid[key] = g
        return self._grid[key]

    def _detect_frame(self, frame: np.ndarray) -> list[Detection]:
        H, W = frame.shape[:2]
        scale = min(1.0, self.max_side / max(H, W))
        img = cv2.resize(frame, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else frame
        h, w = img.shape[:2]
        ph, pw = (h + 31) // 32 * 32, (w + 31) // 32 * 32
        if (ph, pw) != (h, w):
            canvas = np.zeros((ph, pw, 3), dtype=np.uint8)
            canvas[:h, :w] = img
            img = canvas
        if getattr(self.runner, "takes_nhwc_u8", False):
            blob = np.ascontiguousarray(img)[None]            # raw frame; device converts
        else:
            blob = np.empty((1, 3, ph, pw), dtype=np.float32)  # one strided copy with conversion
            blob[0] = img.transpose(2, 0, 1)
        outs = dict(zip(self.out_names, self.runner.run(blob)))
        grids = self._grids(ph, pw)
        boxes, scores, kps = [], [], []
        for s in STRIDES:
            cls = outs[f"cls_{s}"][0, :, 0]
            obj = outs[f"obj_{s}"][0, :, 0]
            sc = np.sqrt(np.clip(cls, 0, 1) * np.clip(obj, 0, 1))
            keep = sc >= self.score_thr
            if not keep.any():
                continue
            bb = outs[f"bbox_{s}"][0][keep]
            xs, ys = grids[s]
            cx = (xs[keep] + bb[:, 0]) * s
            cy = (ys[keep] + bb[:, 1]) * s
            bw = np.exp(bb[:, 2]) * s
            bh = np.exp(bb[:, 3]) * s
            boxes.append(np.stack([cx - bw / 2, cy - bh / 2, bw, bh], axis=1))
            scores.append(sc[keep])
            # 5 landmarks as (dx, dy) offsets from the grid cell, like the box centre
            k = outs[f"kps_{s}"][0][keep].reshape(-1, 5, 2).astype(np.float32)
            k[:, :, 0] = (xs[keep][:, None] + k[:, :, 0]) * s
            k[:, :, 1] = (ys[keep][:, None] + k[:, :, 1]) * s
            kps.append(k)
        if not boxes:
            return []
        boxes = np.concatenate(boxes)
        scores = np.concatenate(scores)
        kps = np.concatenate(kps)
        if len(scores) > self.top_k:
            order = np.argsort(-scores)[: self.top_k]
            boxes, scores, kps = boxes[order], scores[order], kps[order]
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores.astype(float).tolist(), self.score_thr, self.nms_thr)
        out = []
        norm = np.array([W, H], dtype=np.float32)
        for i in np.array(idx).reshape(-1):
            x, y, bw, bh = boxes[i] / scale
            lm = (kps[i] / scale) / norm
            out.append(Detection(BBox.from_pixels(x, y, x + bw, y + bh, W, H), "face", float(scores[i]), self.name,
                                 landmarks=lm))
        return out
