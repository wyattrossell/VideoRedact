"""Does the detector-side brightness/contrast boost (vision/enhance.py) find more?

Runs the shipped detector pipeline on every N-th frame of each clip twice, built with the
enhancement off and on (`AutoDetectOptions.enhance`; the face and plate detectors see the
boosted copy, YOLOX the original), and matches the two detection sets per label (IoU >= 0.4):

  both        found either way (score change reported)
  only_enh    found only on the enhanced frame   (gained: real objects or new false hits)
  only_orig   found only on the original frame   (lost)

Contact sheets of the gained and lost crops go to --out so precision can be judged by eye
(the clips have no face labels), together with each clip's mean grey level and contrast so
gains can be related to how dark the footage is. Results of 2026-11-02 in
bench/results/ENHANCE.md.

Usage: python bench/eval_enhance.py [--clips bench/clips] [--stride 15] [--out <scratch dir>]
          [--labels face,screen,document,phone,license_plate] [--no-multiscale] [--clip 2.0 --grid 8]
          [--raw]   (--raw: enhance every frame for every detector, the first experiment)
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

from videoredact.vision import enhance as enh_mod  # noqa: E402
from videoredact.vision.auto_detect import AutoDetectOptions, build_detector  # noqa: E402
from videoredact.vision.enhance import enhance_for_detection, luma_stats  # noqa: E402

MATCH_IOU = 0.4


def match(a, b):
    """Greedy IoU matching per label; returns (pairs, unmatched_a, unmatched_b)."""
    used_b = set()
    pairs = []
    for i, d in enumerate(a):
        best, best_v = None, MATCH_IOU
        for j, e in enumerate(b):
            if j in used_b or e.label != d.label:
                continue
            v = d.bbox.iou(e.bbox)
            if v > best_v:
                best, best_v = j, v
        if best is not None:
            used_b.add(best)
            pairs.append((i, best))
    ua = [i for i in range(len(a)) if i not in {p[0] for p in pairs}]
    ub = [j for j in range(len(b)) if j not in used_b]
    return pairs, ua, ub


def crop_for(img, d, size=(128, 128)):
    H, W = img.shape[:2]
    x0, y0, x1, y1 = d.bbox.to_pixels(W, H, pad=0.3)
    c = img[y0:y1, x0:x1]
    if c.size == 0:
        return None
    c = cv2.resize(c, size)
    cv2.putText(c, f"{d.label[:6]} {d.score:.2f}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return c


def sheet(crops, path, cols=8, size=(128, 128), limit=320):
    crops = crops[:limit]
    if not crops:
        return
    blank = np.zeros((size[1], size[0], 3), np.uint8)
    rows = [np.hstack(crops[i:i + cols] + [blank] * (cols - len(crops[i:i + cols]))) for i in range(0, len(crops), cols)]
    cv2.imwrite(path, np.vstack(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default="bench/clips")
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--out", required=True, help="scratch folder for the contact sheets and summary.json (not committed)")
    ap.add_argument("--labels", default="face,screen,document,phone,license_plate")
    ap.add_argument("--no-multiscale", action="store_true")
    ap.add_argument("--match", default="", help="only clips whose name contains this")
    ap.add_argument("--clip", type=float, default=None, help="CLAHE clip limit override (0 = stretch only)")
    ap.add_argument("--grid", type=int, default=None, help="CLAHE grid override")
    ap.add_argument("--raw", action="store_true", help="enhance the whole frame for every detector (first experiment)")
    ap.add_argument("--mode", default="always", choices=["auto", "always"], help="boost mode of the enhanced pipeline")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.clip is not None:
        enh_mod.CLIP_LIMIT = a.clip
    if a.grid is not None:
        enh_mod.GRID = a.grid
    labels = a.labels.split(",")
    opts = AutoDetectOptions(labels=labels, multiscale=not a.no_multiscale, face_pad=0.0, enhance="off")
    det = build_detector(opts)
    det_enh = det if a.raw else build_detector(AutoDetectOptions(labels=labels, multiscale=not a.no_multiscale, face_pad=0.0, enhance=a.mode))
    n_boost = 0
    per_label = {}
    gained, lost, both_pairs = [], [], []
    clips = {}
    t_enh = 0.0
    n_frames = 0
    for vid in sorted(glob.glob(os.path.join(a.clips, "*.mp4"))):
        if a.match and a.match not in os.path.basename(vid):
            continue
        cap = cv2.VideoCapture(vid)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        c = dict(frames=0, orig=0, enh=0, both=0, only_enh=0, only_orig=0, luma=[], contrast=[])
        for f in range(0, n, a.stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                break
            m, s = luma_stats(img)
            c["luma"].append(m)
            c["contrast"].append(s)
            n_boost += int(enh_mod.needs_boost(img))
            t0 = time.perf_counter()
            enh = enhance_for_detection(img)
            t_enh += time.perf_counter() - t0
            n_frames += 1
            d_orig = det.detect_one(img)
            d_enh = det.detect_one(enh) if a.raw else det_enh.detect_one(img)
            pairs, ua, ub = match(d_orig, d_enh)
            c["frames"] += 1
            c["orig"] += len(d_orig)
            c["enh"] += len(d_enh)
            c["both"] += len(pairs)
            c["only_orig"] += len(ua)
            c["only_enh"] += len(ub)
            for i, j in pairs:
                pl = per_label.setdefault(d_orig[i].label, dict(both=0, only_enh=0, only_orig=0, dscore=[]))
                pl["both"] += 1
                pl["dscore"].append(d_enh[j].score - d_orig[i].score)
                both_pairs.append((d_orig[i].score, d_enh[j].score))
            for i in ua:
                pl = per_label.setdefault(d_orig[i].label, dict(both=0, only_enh=0, only_orig=0, dscore=[]))
                pl["only_orig"] += 1
                cr = crop_for(img, d_orig[i])
                if cr is not None:
                    lost.append((d_orig[i].score, cr, m, d_orig[i].label))
            for j in ub:
                pl = per_label.setdefault(d_enh[j].label, dict(both=0, only_enh=0, only_orig=0, dscore=[]))
                pl["only_enh"] += 1
                cr = crop_for(img, d_enh[j])          # crop from the ORIGINAL so the eye judges the real picture
                if cr is not None:
                    gained.append((d_enh[j].score, cr, m, d_enh[j].label))
        cap.release()
        c["luma_mean"] = round(float(np.mean(c["luma"])), 1) if c["luma"] else None
        c["luma_min"] = round(float(np.min(c["luma"])), 1) if c["luma"] else None
        c["contrast_mean"] = round(float(np.mean(c["contrast"])), 1) if c["contrast"] else None
        del c["luma"], c["contrast"]
        clips[os.path.basename(vid)] = c
        print(f"{os.path.basename(vid)}: luma {c['luma_mean']} contrast {c['contrast_mean']}: "
              f"orig {c['orig']} enh {c['enh']} both {c['both']} only_enh {c['only_enh']} only_orig {c['only_orig']}", flush=True)
    gained.sort(key=lambda t: -t[0])
    lost.sort(key=lambda t: -t[0])
    sheet([g[1] for g in gained], os.path.join(a.out, "gained_sheet.jpg"))
    sheet([l[1] for l in lost], os.path.join(a.out, "lost_sheet.jpg"))
    for lbl in per_label:
        sheet([g[1] for g in gained if g[3] == lbl], os.path.join(a.out, f"gained_{lbl}.jpg"))
        sheet([l[1] for l in lost if l[3] == lbl], os.path.join(a.out, f"lost_{lbl}.jpg"))
    summary = {
        "stride": a.stride, "labels": labels, "multiscale": not a.no_multiscale, "raw": a.raw, "mode": a.mode,
        "clip": enh_mod.CLIP_LIMIT, "grid": enh_mod.GRID,
        "frames_auto_would_boost": n_boost,
        "frames": n_frames, "enhance_ms_per_frame": round(1000 * t_enh / max(1, n_frames), 2),
        "per_label": {k: dict(both=v["both"], only_enh=v["only_enh"], only_orig=v["only_orig"],
                              mean_score_change=round(float(np.mean(v["dscore"])), 3) if v["dscore"] else None)
                      for k, v in per_label.items()},
        "matched_mean_score": {"orig": round(float(np.mean([p[0] for p in both_pairs])), 3) if both_pairs else None,
                               "enh": round(float(np.mean([p[1] for p in both_pairs])), 3) if both_pairs else None},
        "gained_by_luma": {"dark(<70)": sum(1 for g in gained if g[2] < 70), "mid(70-130)": sum(1 for g in gained if 70 <= g[2] < 130),
                           "bright(>=130)": sum(1 for g in gained if g[2] >= 130)},
        "lost_by_luma": {"dark(<70)": sum(1 for g in lost if g[2] < 70), "mid(70-130)": sum(1 for g in lost if 70 <= g[2] < 130),
                         "bright(>=130)": sum(1 for g in lost if g[2] >= 130)},
        "clips": clips,
    }
    with open(os.path.join(a.out, "summary.json"), "w", newline="\n") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "clips"}, indent=1))


if __name__ == "__main__":
    main()
