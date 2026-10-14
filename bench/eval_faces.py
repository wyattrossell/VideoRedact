"""Compare face detectors on OUR frames (bench/frames).

With ground truth (bench/labels, YOLO) it reports true recall/precision per
detector. Without it (provisional), it builds a *consensus pseudo-ground-truth*:
a face counts if at least `--agree` detectors found it (IoU >= 0.4). Each
detector's recall against that consensus, its unique finds (possible extra true
positives OR false positives) and ms/frame are reported, and contact sheets of
the unique finds are written to bench/results/faces/<detector>.jpg so a human
can judge them. Final choice must use real labels.

Usage: python bench/eval_faces.py [--frames bench/frames] [--limit 300] [--candidates yunet1280,yunet_tiles,scrfd640,centerface]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from videoredact.core.model import BBox  # noqa: E402
from videoredact.vision.detectors import build_face  # noqa: E402

CANDIDATES = {
    "yunet960": lambda: build_face("yunet", max_side=960),
    "yunet1280": lambda: build_face("yunet", max_side=1280),
    "yunet_tiles": lambda: build_face("yunet", max_side=1280, tiles=True),
    "yunet_ms": lambda: build_face("yunet", max_side=1280, multiscale=[1.0, 1.5]),
    "yunet_tiles_ms": lambda: build_face("yunet", max_side=1280, tiles=True, multiscale=[1.0, 1.5]),
    "centerface": lambda: build_face("centerface"),
    "scrfd640": lambda: build_face("scrfd"),
    "scrfd_tiles": lambda: build_face("scrfd", tiles=True),
    "egoblur": lambda: build_face("egoblur"),
}


def iou(a: BBox, b: BBox) -> float:
    return a.iou(b)


def load_yolo(labels_dir: Path, stem: str, classes: list[str]) -> list[BBox] | None:
    clip = stem.rsplit("_f", 1)[0]
    p = labels_dir / clip / f"{stem}.txt"
    if not p.exists():
        return None
    out = []
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5 and classes[int(float(parts[0]))] == "face":
            cx, cy, w, h = map(float, parts[1:5])
            out.append(BBox(cx - w / 2, cy - h / 2, w, h))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="bench/frames")
    ap.add_argument("--labels", default="bench/labels")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--agree", type=int, default=2)
    ap.add_argument("--candidates", default="yunet960,yunet1280,yunet_tiles,yunet_ms,centerface,scrfd640,scrfd_tiles")
    ap.add_argument("--out", default="bench/results/faces")
    a = ap.parse_args()
    frames = sorted(Path(a.frames).rglob("*.jpg"))
    if a.limit:
        frames = frames[:: max(1, len(frames) // a.limit)][: a.limit]
    labels_dir = Path(a.labels)
    classes = ["face", "screen", "phone_screen", "document", "plate", "id_card"]
    cf = labels_dir / "classes.txt"
    if cf.exists():
        classes = [l.strip() for l in cf.read_text().splitlines() if l.strip()]
    dets = {}
    for name in a.candidates.split(","):
        try:
            dets[name] = CANDIDATES[name]()
        except Exception as e:  # noqa: BLE001
            print(f"skip {name}: {e}")
    results = {n: {} for n in dets}
    times = defaultdict(float)
    imgs = {}
    for i, p in enumerate(frames):
        img = cv2.imread(str(p))
        if img is None:
            continue
        imgs[p.stem] = p
        for n, d in dets.items():
            t0 = time.perf_counter()
            results[n][p.stem] = [(x.bbox, x.score) for x in d.detect([img])[0]]
            times[n] += time.perf_counter() - t0
        if i % 50 == 0:
            print(f"{i}/{len(frames)}", flush=True)
    stems = list(imgs)
    # ---- ground truth or consensus ----
    gt = {}
    have_gt = 0
    for s in stems:
        g = load_yolo(labels_dir, s, classes)
        if g is not None:
            gt[s] = g
            have_gt += 1
    mode = "ground-truth" if have_gt else f"consensus(>={a.agree} detectors)"
    if not have_gt:
        for s in stems:
            pool = [(n, b) for n in dets for b, _ in results[n][s]]
            cons = []
            used = [False] * len(pool)
            for i, (n, b) in enumerate(pool):
                if used[i]:
                    continue
                group = {n}
                for j in range(i + 1, len(pool)):
                    if not used[j] and iou(b, pool[j][1]) >= 0.4:
                        group.add(pool[j][0])
                        used[j] = True
                used[i] = True
                if len(group) >= a.agree:
                    cons.append(b)
            gt[s] = cons
    # ---- scores ----
    summary = {"mode": mode, "frames": len(stems), "labelled_frames": have_gt, "detectors": {}}
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for n in dets:
        tp = fp = fn = 0
        unique_tiles = []
        for s in stems:
            g = gt.get(s, [])
            pred = results[n][s]
            matched_g = set()
            for b, sc in pred:
                j = next((k for k, gb in enumerate(g) if k not in matched_g and iou(b, gb) >= 0.4), None)
                if j is None:
                    fp += 1
                    if len(unique_tiles) < 48:
                        img = cv2.imread(str(imgs[s]))
                        H, W = img.shape[:2]
                        x0, y0, x1, y1 = b.to_pixels(W, H, 0.6)
                        crop = img[y0:y1, x0:x1]
                        if crop.size:
                            bx0, by0, bx1, by1 = b.to_pixels(W, H)
                            cv2.rectangle(img, (bx0, by0), (bx1, by1), (0, 255, 0), 2)
                            crop = img[y0:y1, x0:x1]
                            unique_tiles.append(cv2.resize(crop, (160, 160)))
                else:
                    matched_g.add(j)
                    tp += 1
            fn += len(g) - len(matched_g)
        rec = tp / (tp + fn) if (tp + fn) else None
        prec = tp / (tp + fp) if (tp + fp) else None
        summary["detectors"][n] = {"recall": rec, "precision_vs_" + ("gt" if have_gt else "consensus"): prec,
                                   "tp": tp, "fp_or_unique": fp, "fn": fn, "ms_per_frame": round(times[n] / len(stems) * 1000, 1)}
        if unique_tiles:
            while len(unique_tiles) % 8:
                unique_tiles.append(np.zeros_like(unique_tiles[0]))
            sheet = np.vstack([np.hstack(unique_tiles[i:i + 8]) for i in range(0, len(unique_tiles), 8)])
            cv2.imwrite(str(out_dir / f"{n}_unique.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
