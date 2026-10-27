"""Where did we leak, and how much extra did we flag, against reference ground truth?

Takes the tracks saved by bench/run_bench.py (bench/tracks/<stem>.json), the labels derived by
bench/labels_from_reference.py and the original video. Prints, per labelled frame with ground
truth: coverage of each reference box by our rendered boxes (same padding/shape as export), and
writes annotated frames for the leaks (reference box green, ours red) so a person can see why.
Also counts "extra" boxes: ours on frames where the reference had nothing or elsewhere in the
frame (what a reviewer following the same policy would have to delete).

    python bench/leak_report.py bench/reference/tanyard_original.mp4 [--tracks bench/tracks/tanyard_original.json]
           [--labels bench/labels] [--out bench/reference/leaks] [--cover 0.9] [--max-images 30]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from videoredact.core import video_redact  # noqa: E402
from videoredact.core.media import probe  # noqa: E402
from videoredact.core.model import BBox  # noqa: E402
from run_bench import load_gt, load_tracks, coverage  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--tracks", default="")
    ap.add_argument("--labels", default="bench/labels")
    ap.add_argument("--out", default="bench/reference/leaks")
    ap.add_argument("--cover", type=float, default=0.9)
    ap.add_argument("--max-images", type=int, default=30)
    a = ap.parse_args()
    clip = Path(a.video)
    tracks = load_tracks(Path(a.tracks) if a.tracks else Path("bench/tracks") / f"{clip.stem}.json")
    classes = (Path(a.labels) / "classes.txt").read_text(encoding="utf-8").split()
    gt = load_gt(Path(a.labels), clip.stem, classes)
    info = probe(str(clip))
    W, H = info.width, info.height
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    leaks, covered, extra_boxes, extra_frames = [], 0, Counter(), 0
    frames_with_gt = [f for f in sorted(gt.labelled_frames) if gt.boxes.get(f)]
    for f in sorted(gt.labelled_frames):
        ours = []
        for t, bb in video_redact.boxes_at(tracks, f):
            x0, y0, x1, y1 = bb.to_pixels(W, H, t.pad)
            ours.append((t.label, BBox.from_pixels(x0, y0, x1, y1, W, H)))
        reds = [r for _, r in ours]
        gts = gt.boxes.get(f, [])
        for cls, box in gts:
            c = coverage(box, reds)
            if c >= a.cover:
                covered += 1
            else:
                leaks.append((f, cls, box, c, ours))
        gt_boxes = [b for _, b in gts]
        extra_here = 0
        for label, r in ours:
            if not gt_boxes or coverage(r, gt_boxes) < 0.5:
                extra_boxes[label] += 1
                extra_here += 1
        if extra_here:
            extra_frames += 1

    n_gt = covered + len(leaks)
    print(f"{clip.name}: {len(gt.labelled_frames)} labelled frames, {len(frames_with_gt)} with reference boxes, "
          f"{n_gt} reference boxes")
    print(f"covered (>= {a.cover:.0%}): {covered}/{n_gt}  leaks: {len(leaks)}  "
          f"leak frames: {len({f for f, *_ in leaks})}/{len(frames_with_gt)}")
    print(f"extra boxes of ours on labelled frames (reviewer burden): {sum(extra_boxes.values())} "
          f"on {extra_frames}/{len(gt.labelled_frames)} frames; by label: {dict(extra_boxes)}")
    if leaks:
        print("leaks (frame, class, box px, coverage, our boxes on that frame):")
        for f, cls, box, c, ours in leaks[:60]:
            x0, y0, x1, y1 = box.to_pixels(W, H)
            print(f"  f{f:06d} t={f / info.fps:7.1f}s {cls} ({x0},{y0},{x1 - x0}x{y1 - y0}) cov={c:.2f} ours={[l for l, _ in ours]}")

    cap = cv2.VideoCapture(str(clip))
    n_img = 0
    step = max(1, len(leaks) // max(1, a.max_images))
    for i, (f, cls, box, c, ours) in enumerate(leaks):
        if i % step or n_img >= a.max_images:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue
        x0, y0, x1, y1 = box.to_pixels(W, H)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        for label, r in ours:
            rx0, ry0, rx1, ry1 = r.to_pixels(W, H)
            cv2.rectangle(frame, (rx0, ry0), (rx1, ry1), (0, 0, 255), 2)
            cv2.putText(frame, label, (rx0, max(12, ry0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, f"f{f} t={f / info.fps:.1f}s leak {cls} cov={c:.2f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imwrite(str(out / f"leak_f{f:06d}.png"), cv2.resize(frame, (960, 540)))
        n_img += 1
    summary = {"clip": clip.name, "labelled_frames": len(gt.labelled_frames), "frames_with_gt": len(frames_with_gt),
               "gt_boxes": n_gt, "covered": covered, "leaks": len(leaks),
               "leak_frames": len({f for f, *_ in leaks}), "extra_boxes": dict(extra_boxes), "extra_frames": extra_frames}
    (out / f"{clip.stem}_leaks.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"wrote {n_img} leak images to {out}")


if __name__ == "__main__":
    main()
