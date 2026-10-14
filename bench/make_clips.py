"""Cut short test clips out of long recordings (stream copy, no re-encode).

Usage:  python bench/make_clips.py "samples/*.MP4" --out bench/clips --every 300 --length 45 --max 8
Produces bench/clips/<stem>_<mm-ss>.mp4. Clips stay out of git (CJIS footage).
"""
import argparse
import glob
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from videoredact.core.media import ffmpeg_exe, probe  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", help="glob of source videos")
    ap.add_argument("--out", default="bench/clips")
    ap.add_argument("--every", type=float, default=300.0, help="seconds between clip starts")
    ap.add_argument("--length", type=float, default=45.0, help="clip length in seconds")
    ap.add_argument("--offset", type=float, default=60.0, help="skip this many seconds at the start")
    ap.add_argument("--max", type=int, default=8, help="max clips per source")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for src in sorted(glob.glob(a.pattern)):
        info = probe(src)
        stem = Path(src).stem.replace(" ", "_")[:40]
        t = a.offset
        n = 0
        while t + a.length <= info.duration and n < a.max:
            name = out / f"{stem}_{int(t // 60):02d}-{int(t % 60):02d}.mp4"
            if not name.exists():
                # -ss before -i seeks to the previous keyframe; re-encode video quickly so the clip starts exactly at t
                cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t:.3f}", "-i", src,
                       "-t", f"{a.length:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                       "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(name)]
                subprocess.run(cmd, check=True)
            print("clip", name.name)
            t += a.every
            n += 1


if __name__ == "__main__":
    main()
