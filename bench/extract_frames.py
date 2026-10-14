"""Extract frames to label (every N-th frame) from each clip.

Usage:  python bench/extract_frames.py --clips bench/clips --out bench/frames --every 30
Writes bench/frames/<clip_stem>/<clip_stem>_f000123.jpg (frame index is zero-based,
in the clip's own frame numbering). The benchmark evaluates exactly these frames,
so label every extracted frame (empty label file = "nothing sensitive here").
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from videoredact.core.media import FrameReader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="bench/clips")
    ap.add_argument("--out", default="bench/frames")
    ap.add_argument("--every", type=int, default=30)
    ap.add_argument("--quality", type=int, default=92)
    a = ap.parse_args()
    for clip in sorted(Path(a.clips).glob("*.mp4")):
        d = Path(a.out) / clip.stem
        d.mkdir(parents=True, exist_ok=True)
        n = 0
        with FrameReader(clip) as r:
            for idx, f in r.iter_frames():
                if idx % a.every == 0:
                    p = d / f"{clip.stem}_f{idx:06d}.jpg"
                    if not p.exists():
                        cv2.imwrite(str(p), f, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
                    n += 1
        print(f"{clip.name}: {n} frames -> {d}")


if __name__ == "__main__":
    main()
