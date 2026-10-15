"""Compare auto-detect configurations on one clip range: speed, track count
(reviewer burden) and face coverage against a reference detector run on every
15th frame (YuNet@1280, single scale). Coverage against this reference saturates
by construction (same detector family); the informative columns are fps, track
count, head boxes and what the ground-truth benchmark says once labels exist.

Usage: python bench/eval_autodetect.py "samples/clip.MP4" --start 9000 --frames 5400
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from videoredact.core.media import FrameReader  # noqa: E402
from videoredact.core.model import BBox  # noqa: E402
from videoredact.core.verify import _coverage  # noqa: E402
from videoredact.core.video_redact import boxes_at  # noqa: E402
from videoredact.vision.auto_detect import AutoDetectOptions, run_auto_detect  # noqa: E402
from videoredact.vision.detector import FaceDetector  # noqa: E402

CONFIGS = {
    "v0.1.7 (iou, no head, no multiscale)": dict(assoc="iou", head_fallback=False, multiscale=False, face_pad=0.1, lead_in=5, lead_out=5),
    "botsort only": dict(assoc="botsort", head_fallback=False, multiscale=False, face_pad=0.1, lead_in=5, lead_out=5),
    "new default (botsort+head+multiscale+pad25+lead15)": dict(),
    "new default, adaptive stride OFF": dict(adaptive=False),
    "new default, adaptive + scene-cut OFF": dict(adaptive=False, scene_cut=False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--start", type=int, default=9000)
    ap.add_argument("--frames", type=int, default=5400)
    ap.add_argument("--workers", default="1,0", help="comma list; 0 = auto parallel")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    START, N = a.start, a.frames
    fd = FaceDetector(0.5, max_side=1280)
    ref = {}
    with FrameReader(a.clip) as r:
        W, H = r.width, r.height
        r.seek(START)
        for f in range(START, START + N):
            img = r.read()
            if img is None:
                break
            if f % 15 == 0:
                ref[f] = [d.bbox for d in fd.detect(img)]
    n_ref = sum(len(v) for v in ref.values())
    print(f"reference: {n_ref} faces on {len(ref)} sampled frames", flush=True)
    for name, kw in CONFIGS.items():
        for workers in (int(w) for w in a.workers.split(",")):
            with FrameReader(a.clip) as r:
                opts = AutoDetectOptions(labels=["face"], stride=5, start_frame=START, end_frame=START + N, workers=workers, **kw)
                t0 = time.time()
                tracks = run_auto_detect(r, opts, progress=lambda p, m: None)
                dt = time.time() - t0
            covered = 0
            for f, dets in ref.items():
                red = [BBox.from_pixels(*bb.to_pixels(W, H, t.pad), W, H) for t, bb in boxes_at(tracks, f)]
                covered += sum(1 for d in dets if _coverage(d, red) >= 0.6)
            heads = sum(1 for t in tracks if t.label == "head")
            print(f"{name:52s} workers={workers}: {dt:5.1f}s ({N / dt:4.0f} fps) tracks={len(tracks):3d} (heads {heads}) "
                  f"covered-frames={sum(t.total_frames() for t in tracks):6d} face coverage={covered}/{n_ref} = {covered / max(1, n_ref):.0%}", flush=True)


if __name__ == "__main__":
    main()
