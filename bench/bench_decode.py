"""Decode-path benchmark: OpenCV VideoCapture vs FFmpeg raw pipe (CPU and hardware
decoders) at native resolution and with a downscaled proxy.

    python bench/bench_decode.py "bench/clips/x.mp4" [--frames 600]

Reports frames/s for each path. Hardware decoders that are not present on the
machine fail fast and are reported as unavailable.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from videoredact.core.media import FrameReader, ffmpeg_exe, probe  # noqa: E402


def bench_cv2(path: str, n: int) -> float:
    with FrameReader(path) as r:
        t0 = time.perf_counter()
        k = 0
        for _, f in r.iter_frames(0, n):
            k += 1
        return k / (time.perf_counter() - t0)


def bench_pipe(path: str, n: int, W: int, H: int, hw: str | None = None, scale: int | None = None,
               threads: int | None = None) -> tuple[float, str]:
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin"]
    if threads:
        cmd += ["-threads", str(threads)]
    if hw:
        cmd += ["-hwaccel", hw]
        if hw in ("cuda", "qsv", "d3d11va"):
            cmd += ["-hwaccel_output_format", {"cuda": "cuda", "qsv": "qsv", "d3d11va": "d3d11"}[hw]]
    cmd += ["-i", path, "-an", "-frames:v", str(n)]
    vf = []
    if hw == "cuda":
        vf.append(f"scale_cuda={scale}:-2" if scale else "scale_cuda=format=nv12")
        vf.append("hwdownload,format=nv12")
    elif hw == "qsv":
        vf.append(f"scale_qsv=w={scale}:h=-1" if scale else "scale_qsv=format=nv12")
        vf.append("hwdownload,format=nv12")
    elif hw == "d3d11va":
        vf.append("hwdownload,format=nv12")
        if scale:
            vf.append(f"scale={scale}:-2")
    elif scale:
        vf.append(f"scale={scale}:-2:flags=fast_bilinear")
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    w = scale if scale else W
    h = int(round(H * w / W / 2) * 2) if scale else H
    fsz = w * h * 3
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=fsz * 4)
    k = 0
    while True:
        buf = p.stdout.read(fsz)
        if len(buf) < fsz:
            break
        np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        k += 1
    err = p.stderr.read().decode(errors="replace")[-160:].strip()
    p.wait()
    dt = time.perf_counter() - t0
    return (k / dt if k else 0.0), ("" if k == n else f"got {k}/{n}: {err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--frames", type=int, default=600)
    a = ap.parse_args()
    info = probe(a.clip)
    W, H, n = info.width, info.height, min(a.frames, info.frame_count)
    print(f"{Path(a.clip).name}: {W}x{H}, {n} frames")
    print(f"cv2 VideoCapture full-res            : {bench_cv2(a.clip, n):6.0f} fps")
    for label, kw in [("ffmpeg pipe CPU full-res", {}), ("ffmpeg pipe CPU proxy 640", {"scale": 640}),
                      ("ffmpeg pipe CPU proxy 640, 4 thr", {"scale": 640, "threads": 4}),
                      ("ffmpeg cuda (NVDEC) full-res", {"hw": "cuda"}), ("ffmpeg cuda proxy 640", {"hw": "cuda", "scale": 640}),
                      ("ffmpeg qsv (QuickSync) full-res", {"hw": "qsv"}), ("ffmpeg qsv proxy 640", {"hw": "qsv", "scale": 640}),
                      ("ffmpeg d3d11va full-res", {"hw": "d3d11va"}), ("ffmpeg d3d11va proxy 640", {"hw": "d3d11va", "scale": 640})]:
        fps, err = bench_pipe(a.clip, n, W, H, **kw)
        print(f"{label:38s}: {fps:6.0f} fps {('(' + err + ')') if err else ''}")


if __name__ == "__main__":
    main()
