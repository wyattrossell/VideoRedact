"""Pseudo-label our own footage for classes COCO does not have (document / notepad / paper,
licence plate) with an Apache-2.0 open-vocabulary detector (OWLv2, Google), to fine-tune the
fast YOLOX detector on them (train/finetune_yolox.py). Dev machine only; nothing here ships.

    C:\\dev\\venv-train\\Scripts\\python train\\pseudolabel_openvocab.py "Samples\\*.MP4" bench\\reference\\*_original.mp4
        [--every 4] [--out train/pseudo] [--thr 0.22] [--max-per-video 400] [--device cuda]

Writes train/pseudo/frames/<stem>/<stem>_f######.jpg and train/pseudo/labels/<stem>/..txt (YOLO
format, classes in train/pseudo/labels/classes.txt) for every sampled frame, empty files included
(negatives matter). Also writes train/pseudo/review/<stem>_f######.png for the frames with boxes
so a person can skim them and delete bad ones before training (delete the .txt to drop a frame).
Prompts -> class: see PROMPTS. Boxes bigger than --max-frac of the frame are dropped (a "paper"
box covering the whole picture is noise)."""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROMPTS = {
    "document": ["a sheet of paper", "a notepad", "a document", "a clipboard", "a notebook", "a form",
                 "an envelope", "an id card", "a driver's license", "a business card", "a credit card"],
    "plate": ["a license plate"],
}
CLASSES = list(PROMPTS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--every", type=float, default=4.0, help="seconds between sampled frames")
    ap.add_argument("--out", default="train/pseudo")
    ap.add_argument("--thr", type=float, default=0.22)
    ap.add_argument("--max-per-video", type=int, default=400)
    ap.add_argument("--max-frac", type=float, default=0.45, help="drop boxes covering more than this fraction of the frame")
    ap.add_argument("--model", default="google/owlv2-base-patch16-ensemble")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    device = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    proc = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(a.model).eval().to(device)
    if device == "cuda":
        model = model.half()
    queries = [q for qs in PROMPTS.values() for q in qs]
    q2cls = {q: c for c, qs in PROMPTS.items() for q in qs}
    out = Path(a.out)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "review").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")

    files = []
    for pat in a.videos:
        files += [Path(p) for p in glob.glob(pat)]
    print(f"{len(files)} videos, device={device}, model={a.model}")
    totals = {"frames": 0, "boxes": {c: 0 for c in CLASSES}}
    t_all = time.time()
    for vid in files:
        stem = vid.stem.strip().replace(" ", "_")
        cap = cv2.VideoCapture(str(vid))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dur = n / fps
        every = max(a.every, dur / a.max_per_video)
        (out / "labels" / stem).mkdir(exist_ok=True)
        (out / "frames" / stem).mkdir(exist_ok=True)
        k = 0
        t0 = time.time()
        nb = 0
        while True:
            t = k * every
            if t >= dur:
                break
            fi = int(round(t * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                break
            H, W = frame.shape[:2]
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            with torch.no_grad():
                inputs = proc(text=[queries], images=pil, return_tensors="pt").to(device)
                if device == "cuda":
                    inputs["pixel_values"] = inputs["pixel_values"].half()
                outputs = model(**inputs)
                target = torch.tensor([[H, W]])
                try:
                    res = proc.post_process_grounded_object_detection(outputs, threshold=a.thr, target_sizes=target, text_labels=[queries])[0]
                    labels = res.get("text_labels") or [queries[int(i)] for i in res["labels"]]
                except (AttributeError, TypeError):
                    res = proc.post_process_object_detection(outputs, threshold=a.thr, target_sizes=target)[0]
                    labels = [queries[int(i)] for i in res["labels"]]
            dets = []
            for lab, sc, (x0, y0, x1, y1) in zip(labels, res["scores"].tolist(), res["boxes"].tolist()):
                x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
                w, h = x1 - x0, y1 - y0
                if w < 8 or h < 8 or (w * h) > a.max_frac * W * H:
                    continue
                dets.append((q2cls[lab], float(sc), x0, y0, w, h))
            # NMS across prompts of the same class
            keep = []
            for d in sorted(dets, key=lambda d: -d[1]):
                c, s, x0, y0, w, h = d
                dup = False
                for kd in keep:
                    if kd[0] != c:
                        continue
                    ix0, iy0 = max(x0, kd[2]), max(y0, kd[3])
                    ix1, iy1 = min(x0 + w, kd[2] + kd[4]), min(y0 + h, kd[3] + kd[5])
                    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
                    if inter / (w * h + kd[4] * kd[5] - inter + 1e-6) > 0.5:
                        dup = True
                        break
                if not dup:
                    keep.append(d)
            name = f"{stem}_f{fi:06d}"
            cv2.imwrite(str(out / "frames" / stem / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            lines = [f"{CLASSES.index(c)} {(x0 + w / 2) / W:.6f} {(y0 + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}" for c, s, x0, y0, w, h in keep]
            (out / "labels" / stem / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            if keep:
                vis = frame.copy()
                for c, s, x0, y0, w, h in keep:
                    cv2.rectangle(vis, (int(x0), int(y0)), (int(x0 + w), int(y0 + h)), (0, 255, 0), 2)
                    cv2.putText(vis, f"{c} {s:.2f}", (int(x0), max(12, int(y0) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imwrite(str(out / "review" / f"{name}.png"), cv2.resize(vis, (960, 540)))
                nb += len(keep)
                for c, *_ in keep:
                    totals["boxes"][c] += 1
            totals["frames"] += 1
            k += 1
        print(f"{vid.name[:50]:50s} {k:4d} frames, {nb:4d} boxes, {time.time() - t0:5.0f}s", flush=True)
    totals["elapsed_s"] = round(time.time() - t_all)
    (out / "summary.json").write_text(json.dumps(totals, indent=1), encoding="utf-8")
    print(json.dumps(totals))


if __name__ == "__main__":
    main()
