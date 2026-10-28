"""Evaluate open-vocabulary detectors (Apache-2.0: OWL-ViT / OWLv2 from Google via transformers)
for the "document / notepad / paper" gap that COCO-class detectors leave (YOLOX never fires
"book" on a notepad in body-cam footage). Dev/eval only: runs in C:\\dev\\venv-eval (torch CPU);
nothing here ships. Downloads the model from Hugging Face on first use (dev machine only).

    C:\\dev\\venv-eval\\Scripts\\python bench\\eval_openvocab.py VIDEO --times 666,684,120,900
        [--model google/owlvit-base-patch32] [--queries "a notepad,a sheet of paper,..."] [--thr 0.1]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

DEFAULT_QUERIES = ["a notepad", "a sheet of paper", "a document", "a clipboard", "a notebook",
                   "a computer screen", "a mobile phone", "a license plate", "a credit card", "an id card"]


def enhance(bgr: np.ndarray) -> np.ndarray:
    """Lift dark body-cam frames: CLAHE on L (LAB) then gamma 0.6."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    lut = np.array([((i / 255.0) ** 0.6) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(out, lut)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--times", default="666,669,678,684,120,900,1500,2200")
    ap.add_argument("--model", default="google/owlvit-base-patch32")
    ap.add_argument("--queries", default=",".join(DEFAULT_QUERIES))
    ap.add_argument("--thr", type=float, default=0.10)
    ap.add_argument("--size", type=int, default=0, help="resize longer side before inference (0 = model default)")
    ap.add_argument("--out", default="bench/reference/preview_openvocab")
    ap.add_argument("--device", default="cpu", help="cpu or cuda (C:/dev/venv-train has the CUDA torch)")
    ap.add_argument("--enhance", action="store_true",
                    help="CLAHE + gamma 0.6 before inference (does the teacher see dark paper better?)")
    a = ap.parse_args()

    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    torch.set_num_threads(8)
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(a.model).eval().to(a.device)
    print(f"loaded {a.model} in {time.time() - t0:.0f}s")
    queries = [q.strip() for q in a.queries.split(",") if q.strip()]
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    for t in [float(x) for x in a.times.split(",")]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, frame = cap.read()
        if not ok:
            continue
        img = frame
        if a.size:
            s = a.size / max(frame.shape[:2])
            img = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if a.enhance:
            img = enhance(img)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil = Image.fromarray(rgb)
        t1 = time.time()
        with torch.no_grad():
            inputs = proc(text=[queries], images=pil, return_tensors="pt").to(a.device)
            outputs = model(**inputs)
            target = torch.tensor([pil.size[::-1]], device=a.device)
            try:
                res = proc.post_process_grounded_object_detection(outputs, threshold=a.thr, target_sizes=target, text_labels=[queries])[0]
                labels = res.get("text_labels") or [queries[int(i)] for i in res["labels"]]
            except (AttributeError, TypeError):
                res = proc.post_process_object_detection(outputs, threshold=a.thr, target_sizes=target)[0]
                labels = [queries[int(i)] for i in res["labels"]]
        dt = time.time() - t1
        dets = sorted(zip(labels, res["scores"].tolist(), res["boxes"].tolist()), key=lambda x: -x[1])
        sx = frame.shape[1] / img.shape[1]
        print(f"t={t:6.0f}s  {dt:5.2f}s  " + ", ".join(f"{l}:{s:.2f}" for l, s, _ in dets[:8]))
        vis = frame.copy()
        for l, s, (x0, y0, x1, y1) in dets:
            x0, y0, x1, y1 = [int(v * sx) for v in (x0, y0, x1, y1)]
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(vis, f"{l} {s:.2f}", (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(str(out_dir / f"{Path(a.model).name}_t{int(t):05d}.png"), cv2.resize(vis, (960, 540)))


if __name__ == "__main__":
    main()
