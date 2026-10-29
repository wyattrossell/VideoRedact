"""The auto-detect proxy pipe must never feed frozen frames (2026-10-29: Intel QuickSync returned
the first frame of a 1080p clip 399 times through scale_qsv + hwdownload, so every auto-detect box
on 1080p sources was static while the export looked fine)."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from videoredact.core.media import FrameReader, ffmpeg_exe
from videoredact.core.streams import FFmpegWriter, FrameProducer, GUARD_FRAMES, _frames_frozen

W, H, N = 1920, 1080, 40


@pytest.fixture(scope="module")
def clip_1080p(tmp_path_factory) -> Path:
    """A 1080p clip with a moving square, so consecutive frames always differ."""
    out = tmp_path_factory.mktemp("decode") / "moving_1080p.mp4"
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{W}x{H}", "-r", "30", "-i", "pipe:0", "-c:v", "libx264", "-preset", "ultrafast",
           "-pix_fmt", "yuv420p", str(out)]
    w = FFmpegWriter(cmd)
    for i in range(N):
        f = np.full((H, W, 3), 40, np.uint8)
        x = 100 + i * 30
        cv2.rectangle(f, (x, 300), (x + 300, 700), (0, 200, 255), -1)
        w.put(f)
    w.close()
    return out


def test_frames_frozen_helper():
    a = np.zeros((8, 8, 3), np.uint8)
    b = a.copy()
    b[0, 0] = 255
    assert _frames_frozen([a, a.copy(), a.copy()])
    assert not _frames_frozen([a, b, a.copy()])
    assert not _frames_frozen([a])


def test_proxy_auto_frames_change_and_match_source(clip_1080p):
    with FrameProducer(str(clip_1080p), 0, N, proxy_width=1280, hwaccel="auto") as p:
        frames = list(p)
    assert p.engine.startswith("ffmpeg-pipe"), p.engine
    assert [i for i, _ in frames] == list(range(N))
    assert frames[0][1].shape == (720, 1280, 3)
    assert not _frames_frozen([f for _, f in frames])
    with FrameReader(clip_1080p) as r:
        ref = cv2.resize(r.read_at(20), (1280, 720))
    assert float(np.mean(cv2.absdiff(ref, frames[20][1]))) < 12


def test_frozen_hardware_decoder_is_rejected(clip_1080p, monkeypatch):
    """A decoder whose first GUARD_FRAMES frames are identical is dropped and the next one used;
    none of its frames reach the consumer."""
    orig = FrameProducer._pipe_cmd

    def fake_cmd(self, hw):
        if hw is None:
            return orig(self, hw)
        # a 'hardware decoder' that emits one static colour: what a frozen QuickSync pipe looks like
        return [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", "-f", "lavfi",
                "-i", f"color=c=red:size={self.w}x{self.h}:rate=30", "-frames:v", str(self.end - self.start),
                "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]

    monkeypatch.setattr(FrameProducer, "_pipe_cmd", fake_cmd)
    with FrameProducer(str(clip_1080p), 0, N, proxy_width=1280, hwaccel="auto") as p:
        frames = list(p)
    assert p.rejected == ["d3d11va"]
    assert p.engine == "ffmpeg-pipe/cpu"
    assert len(frames) == N and GUARD_FRAMES < N
    assert not any(np.array_equal(f[:, :, 2], np.full((720, 1280), 255, np.uint8)) for _, f in frames)
    assert not _frames_frozen([f for _, f in frames])
