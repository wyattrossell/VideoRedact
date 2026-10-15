import os
from pathlib import Path

import numpy as np
import pytest

from videoredact.core.media import FrameReader, ffmpeg_exe, probe
from videoredact.core.streams import FFmpegWriter, FrameProducer

SAMPLE = Path("tests/data/sample.mp4")
pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="sample clip missing (run scripts/make_sample.py)")


def test_producer_cv2_matches_reader():
    with FrameReader(SAMPLE) as r:
        ref = [f.copy() for _, f in r.iter_frames(10, 40)]
    with FrameProducer(str(SAMPLE), 10, 40) as p:
        got = list(p)
    assert p.engine == "opencv"
    assert [i for i, _ in got] == list(range(10, 40))
    assert all(np.array_equal(a, b) for (_, a), b in zip(got, ref))


def test_producer_step_and_proxy_pipe():
    with FrameProducer(str(SAMPLE), 0, 60, step=3) as p:
        idx = [i for i, _ in p]
    assert idx == list(range(0, 60, 3))
    # proxy: 640 -> 320 wide through the FFmpeg pipe
    with FrameProducer(str(SAMPLE), 30, 60, proxy_width=320, hwaccel=None) as p:
        frames = list(p)
    assert p.engine.startswith("ffmpeg-pipe")
    assert len(frames) == 30 and frames[0][1].shape == (180, 320, 3) and frames[0][0] == 30
    with FrameReader(SAMPLE) as r:
        full = r.read_at(45)
    import cv2
    small = cv2.resize(full, (320, 180))
    diff = np.abs(small.astype(int) - frames[15][1].astype(int)).mean()
    assert diff < 12  # same frame content up to scaler/decoder differences


def test_ffmpeg_writer_encodes(tmp_path):
    out = tmp_path / "w.mp4"
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", "64x48", "-r", "10", "-i", "pipe:0", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(out)]
    w = FFmpegWriter(cmd)
    for i in range(20):
        w.put(np.full((48, 64, 3), i * 10, np.uint8))
    w.close()
    info = probe(str(out))
    assert info.has_video and info.frame_count == 20
