"""Manual-box tracking evaluation: how much of an object's visible life does one drawn box cover?

Cases are confident auto-detected face tracks saved by run_bench.py (bench/tracks/<stem>.json;
the precise profile makes them ~all real). For each case the box is "drawn" at the track's
middle keyframe and the single-object tracker runs the way the app does (both directions,
8 s lost limit). Each case runs twice: with the face detector (label "face") and without
(label "other", which is what a licence plate or tattoo gets today).

Metrics per case / engine:
  coverage     share of the auto track's frames the tracked box covers (recall proxy)
  overrun      tracked frames outside the auto track's span
  unsupported  of the overrun frames sampled every 10th: those where no auto face track overlaps
               the box AND the face detector (thr 0.3, full resolution) finds no face overlapping
               it either - a proxy for boxes left on the background
  spans        visibility spans in the result (fragmentation)
  secs         wall time

Usage:
  python bench/track_eval.py --engine anchored --out bench/results/track_eval_anchored.json
  python bench/track_eval.py --engine classic  --out bench/results/track_eval_classic.json
  python bench/track_eval.py --compare bench/results/track_eval_classic.json bench/results/track_eval_anchored.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from videoredact.core.media import FrameReader  # noqa: E402
from videoredact.core.model import BBox  # noqa: E402
from videoredact.paths import DEFAULT_SETTINGS as S  # noqa: E402
from videoredact.vision.detector import CombinedDetector  # noqa: E402
from videoredact.vision.tracker import ObjectTracker, TrackOptions  # noqa: E402


def load_cases(tracks_dir: str, clips_dir: str, min_frames: int, max_cases: int, cls: str = "face", seed: int = 1):
    """Confident auto tracks of class `cls` (face, or a rigid object class such as phone) as cases."""
    cases = []
    for p in sorted(glob.glob(os.path.join(tracks_dir, "*.json"))):
        stem = os.path.basename(p)[:-5]
        vid = os.path.join(clips_dir, stem + ".mp4")
        if not os.path.exists(vid):
            continue
        tracks = json.load(open(p))
        same = [t for t in tracks if t["label"] == cls]
        for t in same:
            if len(t["spans"]) != 1:
                continue
            s = t["spans"][0]
            if s["end_frame"] - s["start_frame"] + 1 < min_frames:
                continue
            keys = sorted(int(k) for k in s["keyframes"])
            mid = keys[len(keys) // 2]
            cases.append(dict(vid=vid, stem=stem, cls=cls, start=s["start_frame"], end=s["end_frame"], mid=mid,
                              box=s["keyframes"][str(mid)], all_same=same))
    random.Random(seed).shuffle(cases)
    cases.sort(key=lambda c: c["box"][3])          # by box height, small to large
    if max_cases and len(cases) > max_cases:
        step = len(cases) / max_cases
        cases = [cases[int(i * step)] for i in range(max_cases)]
    return cases


def _overlaps_auto(tracks, frame: int, box: BBox, thr: float = 0.3) -> bool:
    for t in tracks:
        for s in t["spans"]:
            if s["start_frame"] <= frame <= s["end_frame"]:
                keys = sorted(int(k) for k in s["keyframes"])
                k = min(keys, key=lambda q: abs(q - frame))
                if box.iou(BBox(*s["keyframes"][str(k)])) >= thr:
                    return True
    return False


SHEET: list = []      # (crop, caption) of unsupported overrun frames, written by --sheet


def run_case(reader, c, engine: str, label: str, det, probe_det, fps: float):
    """`label` is the class the box is tracked as: the case's own class (with its detector) or
    "other" (no detector: what a licence plate or tattoo gets today)."""
    opts = TrackOptions(engine=engine, step=2, backward=True, tracker="vit", lost_limit_s=8.0,
                        end_frame=min(reader.frame_count - 1, c["end"] + int(5 * fps)))
    t0 = time.time()
    track = ObjectTracker(reader, det if label != "other" else None, opts).track(c["mid"], BBox(*c["box"]), label=label)
    secs = time.time() - t0
    n_auto = c["end"] - c["start"] + 1
    covered = sum(1 for f in range(c["start"], c["end"] + 1) if track.bbox_at(f) is not None)
    frames = sorted({f for s in track.spans for f in range(s.start_frame, s.end_frame + 1)})
    over = [f for f in frames if f < c["start"] or f > c["end"]]
    unsupported = checked = 0
    for f in over[::10]:
        b = track.bbox_at(f)
        if b is None:
            continue
        checked += 1
        if _overlaps_auto(c["all_same"], f, b):
            continue
        img = reader.read_at(f)
        if img is None:
            continue
        if any(b.iou(d.bbox) >= 0.3 for d in probe_det.detect(img) if d.label == c["cls"]):
            continue
        unsupported += 1
        if len(SHEET) < 400:
            import cv2
            H, W = img.shape[:2]
            x0, y0, x1, y1 = b.to_pixels(W, H, pad=0.6)
            crop = img[y0:y1, x0:x1]
            if crop.size:
                crop = cv2.resize(crop, (160, 160))
                cv2.putText(crop, f"{c['stem'][-5:]} {f}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                SHEET.append(crop)
    return dict(stem=c["stem"][-22:], mid=c["mid"], cls=c["cls"], label=label, engine=engine,
                obj_px=round(c["box"][3] * reader.height),
                auto=n_auto, covered=covered, cov=round(covered / n_auto, 3), tracked=track.total_frames(),
                spans=len(track.spans), overrun=len(over), checked=checked, unsupported=unsupported, secs=round(secs, 1))


def summarise(rows):
    out = {}
    for key in sorted({(r["cls"], r["label"]) for r in rows}):
        rs = [r for r in rows if (r["cls"], r["label"]) == key]
        cov = sum(r["covered"] for r in rs) / max(1, sum(r["auto"] for r in rs))
        name = f"{key[0]} as {key[1]}" if key[1] != key[0] else f"{key[0]} (detector)"
        out[name] = dict(cases=len(rs), coverage=round(cov, 3), full=sum(1 for r in rs if r["cov"] >= 0.9),
                         spans=sum(r["spans"] for r in rs), overrun=sum(r["overrun"] for r in rs),
                         checked=sum(r["checked"] for r in rs), unsupported=sum(r["unsupported"] for r in rs),
                         secs=round(sum(r["secs"] for r in rs), 1))
    return out


def fmt(summary: dict) -> str:
    lines = []
    for label, s in summary.items():
        uns = f"{s['unsupported']}/{s['checked']}" if s["checked"] else "0/0"
        lines.append(f"  {label:18}: {s['cases']} cases, coverage {s['coverage']:.1%}, {s['full']}/{s['cases']} >= 90 %, "
                     f"{s['spans']} spans, overrun {s['overrun']} frames (unsupported {uns} of the sampled), {s['secs']} s")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default="anchored", choices=["anchored", "classic"])
    ap.add_argument("--tracks", default="bench/tracks")
    ap.add_argument("--clips", default="bench/clips")
    ap.add_argument("--max-cases", type=int, default=24, help="per case class")
    ap.add_argument("--min-frames", type=int, default=90)
    ap.add_argument("--classes", default="face,phone", help="auto-track classes used as cases (phone = rigid object proxy for plates)")
    ap.add_argument("--labels", default="own,other", help="own = tracked as its class with the detector; other = no detector")
    ap.add_argument("--out", default="")
    ap.add_argument("--sheet", default="", help="write a contact sheet (jpg) of unsupported overrun frames here")
    ap.add_argument("--compare", nargs=2, metavar="JSON", help="print a comparison of two result files and exit")
    a = ap.parse_args()
    if a.compare:
        for p in a.compare:
            d = json.load(open(p))
            print(f"{p} (engine {d['engine']}):")
            print(fmt(d["summary"]))
        return
    cases = []
    for cls in a.classes.split(","):
        cases += load_cases(a.tracks, a.clips, a.min_frames, a.max_cases, cls=cls)
    print(f"{len(cases)} cases ({a.classes}), engine {a.engine}", flush=True)
    dets, probes = {}, {}

    def det_for(cls):
        if cls not in dets:
            dets[cls] = CombinedDetector([cls], S["face_conf"], S["detect_conf"], S.get("detector_model", "yolox_tiny"), 0,
                                         face_model=S.get("face_detector", "yunet"))
            probes[cls] = CombinedDetector([cls], 0.3, 0.3, "yolox_tiny", 0, face_model="yunet")
        return dets[cls], probes[cls]
    rows = []
    by_vid = defaultdict(list)
    for c in cases:
        by_vid[c["vid"]].append(c)
    for vid, cs in by_vid.items():
        with FrameReader(vid) as reader:
            for c in cs:
                det, probe = det_for(c["cls"])
                for label in a.labels.split(","):
                    label = c["cls"] if label == "own" else label
                    r = run_case(reader, c, a.engine, label, det, probe, reader.fps)
                    rows.append(r)
                    print(f"{r['stem']} f{r['mid']:>5} {c['cls']:5} as {label:5} {r['obj_px']:3}px | auto {r['auto']:4} covered {r['cov']:5.0%} "
                          f"spans {r['spans']:2} overrun {r['overrun']:4} unsupported {r['unsupported']}/{r['checked']} {r['secs']}s", flush=True)
    summary = summarise(rows)
    print("==== summary ====")
    print(fmt(summary))
    if a.sheet and SHEET:
        import cv2
        import numpy as np
        cols = 10
        pad = [np.zeros((160, 160, 3), np.uint8)] * ((cols - len(SHEET) % cols) % cols)
        tiles = SHEET + pad
        cv2.imwrite(a.sheet, np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))
        print("wrote", a.sheet, len(SHEET), "unsupported frames")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", newline="\n") as f:
            json.dump(dict(engine=a.engine, cases=len(cases), summary=summary, rows=rows), f, indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
