"""Evaluate a fine-tuned YOLOX checkpoint per class on the val split, and run it on chosen frames.

    C:\\dev\\venv-train\\Scripts\\python train\\eval_ckpt.py --exp train\\runs\\exp_yolox_tiny.py
        --ckpt C:\\dev\\vr-train\\runs\\yolox_tiny_videoredact\\best_ckpt.pth --data C:\\dev\\vr-train\\data
        [--size 416] [--video bench\\reference\\tanyard_original.mp4 --times 666,669,678,684] [--thr 0.1]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, os.getcwd())


def load_exp(path: str):
    spec = importlib.util.spec_from_file_location("exp_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Exp()


def preprocess(img: np.ndarray, size: int):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    res = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    padded[: res.shape[0], : res.shape[1]] = res
    x = torch.from_numpy(padded.transpose(2, 0, 1)).float().unsqueeze(0)
    return x, r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="C:/dev/vr-train/data")
    ap.add_argument("--size", type=int, default=0)
    ap.add_argument("--video", default="")
    ap.add_argument("--times", default="666,669,678,684,120,900")
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--max-val", type=int, default=0)
    ap.add_argument("--split", default="val", help="val or train (train = can the model fit its own labels?)")
    a = ap.parse_args()
    from yolox.utils import postprocess
    exp = load_exp(a.exp)
    size = a.size or exp.test_size[0]
    classes = json.loads((Path(a.data) / "classes.json").read_text())
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = exp.get_model()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.to(dev).eval()
    if hasattr(model, "head"):
        model.head.decode_in_inference = True

    def run(img):
        x, r = preprocess(img, size)
        with torch.no_grad():
            out = model(x.to(dev))
            out = postprocess(out, len(classes), 0.01, 0.65, class_agnostic=False)[0]
        dets = []
        if out is not None:
            for row in out.cpu().numpy():
                x0, y0, x1, y1 = row[:4] / r
                score = float(row[4] * row[5])
                dets.append((int(row[6]), score, x0, y0, x1, y1))
        return dets

    # ---- per-class COCO AP on val ----
    ann_file = Path(a.data) / "annotations" / f"instances_{a.split}.json"
    if ann_file.exists():
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        gt = COCO(str(ann_file))
        results = []
        img_ids = sorted(gt.getImgIds())
        if a.max_val:
            img_ids = img_ids[: a.max_val]
        for iid in img_ids:
            info = gt.loadImgs(iid)[0]
            img = cv2.imread(str(Path(a.data) / f"{a.split}2017" / info["file_name"]))
            if img is None:
                continue
            for c, s, x0, y0, x1, y1 in run(img):
                results.append({"image_id": iid, "category_id": c + 1, "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)], "score": s})
        print(f"{a.split}: {len(img_ids)} images, {len(results)} detections above 0.01")
        if results:
            dt = gt.loadRes(results)
            for cid, name in [(i + 1, n) for i, n in enumerate(classes)]:
                ev = COCOeval(gt, dt, "bbox")
                ev.params.catIds = [cid]
                ev.params.imgIds = img_ids
                ev.evaluate(); ev.accumulate()
                import io, contextlib
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    ev.summarize()
                ap50 = ev.stats[1]; ap_s = ev.stats[3]; ap_m = ev.stats[4]; ap_l = ev.stats[5]; ar100 = ev.stats[8]
                n_gt = len(gt.getAnnIds(catIds=[cid]))
                print(f"  {name:10s} gt={n_gt:4d}  AP50={ap50:.3f}  AP(small/med/large)={ap_s:.3f}/{ap_m:.3f}/{ap_l:.3f}  AR@100={ar100:.3f}")

    # ---- chosen frames ----
    if a.video:
        cap = cv2.VideoCapture(a.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        for t in [float(x) for x in a.times.split(",")]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
            ok, img = cap.read()
            if not ok:
                continue
            dets = [d for d in run(img) if d[1] >= a.thr]
            print(f"t={t:6.0f}s: " + ", ".join(f"{classes[c]} {s:.2f} ({int(x0)},{int(y0)},{int(x1 - x0)}x{int(y1 - y0)})" for c, s, x0, y0, x1, y1 in sorted(dets, key=lambda d: -d[1])[:6]))


if __name__ == "__main__":
    main()
