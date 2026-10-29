"""Licence-plate detector evaluation on our clips (Phase 2 of the tracking rebuild).

Runs one or more plate detectors on every N-th frame of each clip, times them, and writes a
contact sheet of detection crops per detector so precision can be judged by eye (the clips
have no plate ground truth yet). Candidates (see the 2026-10-29 licence survey in
bench/results/PLATES.md):

  rtdetr   Topurrra/rtdetr-license-plate-detection-onnx: RT-DETRv2-R18 fine-tuned on Open Images
           "Vehicle registration plate" (Apache-2.0 code+weights, CC-BY data). Fixed 1x3x640x640,
           RGB /255, outputs logits (1,300,1) + boxes (1,300,4) cxcywh normalized; scores are
           compressed (top box ~0.1), threshold ~0.05.
  yolov9t  open-image-models yolo-v9-t-384 end2end (MIT repo; weights/data terms unclear, so
           benchmark only, never shipped). Input 1x3x384x384 RGB /255 letterboxed, output
           (N,7) [batch, x1, y1, x2, y2, class, score] in letterboxed pixels.

Usage: python bench/eval_plates.py --models rtdetr=path/to/plate_rtdetr.onnx yolov9t=path/to/yolo-v9-t-384.onnx
          [--clips bench/clips] [--stride 15] [--out bench/results/plates]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RTDETRPlate:
    def __init__(self, path: str, thr: float = 0.05, threads: int = 0):
        import onnxruntime as ort
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.thr = thr
        self.name = "rtdetr"

    def detect(self, frame: np.ndarray):
        H, W = frame.shape[:2]
        img = cv2.cvtColor(cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        blob = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        outs = self.sess.run(None, {self.inp: blob})
        logits, boxes = None, None
        for o in outs:
            if o.ndim == 3 and o.shape[-1] == 4:
                boxes = o[0]
            elif o.ndim == 3:
                logits = o[0]
        scores = 1.0 / (1.0 + np.exp(-logits.max(axis=-1)))
        dets = []
        for s, (cx, cy, w, h) in zip(scores, boxes):
            if s < self.thr:
                continue
            dets.append(((cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H, float(s)))
        return dets


class YoloV9Plate:
    def __init__(self, path: str, thr: float = 0.3, threads: int = 0):
        import onnxruntime as ort
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        i = self.sess.get_inputs()[0]
        self.inp, self.size = i.name, int(i.shape[-1])
        self.thr = thr
        self.name = "yolov9t"

    def detect(self, frame: np.ndarray):
        H, W = frame.shape[:2]
        S = self.size
        r = min(S / W, S / H)
        nw, nh = int(round(W * r)), int(round(H * r))
        canvas = np.full((S, S, 3), 114, np.uint8)
        dx, dy = (S - nw) // 2, (S - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        blob = (cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        out = self.sess.run(None, {self.inp: blob})[0]
        out = out.reshape(-1, out.shape[-1])
        dets = []
        for row in out:
            _, x1, y1, x2, y2, _, s = row[:7]
            if s < self.thr:
                continue
            dets.append(((x1 - dx) / r, (y1 - dy) / r, (x2 - x1) / r, (y2 - y1) / r, float(s)))
        return dets


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="name=path (rtdetr=..., yolov9t=...)")
    ap.add_argument("--clips", default="bench/clips")
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--out", default="bench/results/plates")
    ap.add_argument("--thr", type=float, default=0.0, help="override the detector's default threshold")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dets = {}
    for spec in a.models:
        name, path = spec.split("=", 1)
        if name == "rtdetr":
            dets[name] = RTDETRPlate(path, a.thr or 0.05)
        elif name == "yolov9t":
            dets[name] = YoloV9Plate(path, a.thr or 0.3)
        else:
            raise SystemExit(f"unknown model {name}")
    results = {n: dict(frames=0, dets=0, ms=0.0, crops=[]) for n in dets}
    for vid in sorted(glob.glob(os.path.join(a.clips, "*.mp4"))):
        cap = cv2.VideoCapture(vid)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for f in range(0, n, a.stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                break
            for name, d in dets.items():
                t0 = time.perf_counter()
                out = d.detect(img)
                r = results[name]
                r["ms"] += (time.perf_counter() - t0) * 1000
                r["frames"] += 1
                r["dets"] += len(out)
                for (x, y, w, h, s) in out:
                    x0, y0 = int(max(0, x - w * 0.3)), int(max(0, y - h * 0.5))
                    x1, y1 = int(min(img.shape[1], x + w * 1.3)), int(min(img.shape[0], y + h * 1.5))
                    crop = img[y0:y1, x0:x1]
                    if crop.size:
                        crop = cv2.resize(crop, (160, 80))
                        cv2.putText(crop, f"{s:.2f}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                        r["crops"].append((s, crop, os.path.basename(vid), f))
        cap.release()
        print(f"{os.path.basename(vid)}: " + ", ".join(f"{n} {results[n]['dets']} dets" for n in dets), flush=True)
    summary = {}
    for name, r in results.items():
        crops = sorted(r["crops"], key=lambda c: -c[0])[:240]
        if crops:
            rows = [np.hstack([c[1] for c in crops[i:i + 8]] + [np.zeros((80, 160, 3), np.uint8)] * (8 - len(crops[i:i + 8])))
                    for i in range(0, len(crops), 8)]
            cv2.imwrite(os.path.join(a.out, f"{name}_sheet.jpg"), np.vstack(rows))
        summary[name] = dict(frames=r["frames"], detections=r["dets"], ms_per_frame=round(r["ms"] / max(1, r["frames"]), 1),
                             dets_per_frame=round(r["dets"] / max(1, r["frames"]), 3),
                             by_clip=sorted({c[2] for c in r["crops"]}))
        print(f"{name}: {r['frames']} frames, {r['dets']} detections, {summary[name]['ms_per_frame']} ms/frame")
    with open(os.path.join(a.out, "summary.json"), "w", newline="\n") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
