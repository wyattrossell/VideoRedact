"""Media access: FFmpeg discovery, probing, audio decoding, frame reading.

FFmpeg resolution order:
  1. VIDEOREDACT_FFMPEG environment variable
  2. bin/ffmpeg.exe next to the executable / repo root (installer bundles an LGPL build)
  3. ffmpeg on PATH
  4. the binary shipped with the imageio-ffmpeg wheel (dev fallback)
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterator, Optional

import cv2
import numpy as np

from .model import MediaInfo

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".wmv", ".m4v", ".mpg", ".mpeg", ".webm", ".3gp", ".ts", ".mts", ".m2ts", ".flv", ".asf"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".wma", ".opus", ".amr", ".aif", ".aiff"}
MEDIA_FILTER = ("Media files (*" + " *".join(sorted(VIDEO_EXTS | AUDIO_EXTS)) + ");;"
                "Video (*" + " *".join(sorted(VIDEO_EXTS)) + ");;"
                "Audio (*" + " *".join(sorted(AUDIO_EXTS)) + ");;All files (*)")

_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def app_root() -> Path:
    """Directory holding bundled resources (PyInstaller) or the repo root."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parents[2]


def ffmpeg_exe() -> str:
    env = os.environ.get("VIDEOREDACT_FFMPEG")
    if env and Path(env).exists():
        return env
    bundled = app_root() / "bin" / ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
    if bundled.exists():
        return str(bundled)
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # pragma: no cover
        raise RuntimeError("FFmpeg not found. Set VIDEOREDACT_FFMPEG or install ffmpeg.") from e


def run_ffmpeg(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run([ffmpeg_exe(), "-hide_banner", *args], capture_output=True,
                          creationflags=_CREATE_NO_WINDOW, **kw)


def sha256_file(path: str | Path, progress: Optional[Callable[[float], None]] = None) -> str:
    h = hashlib.sha256()
    p = Path(path)
    total = p.stat().st_size or 1
    done = 0
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
            done += len(chunk)
            if progress:
                progress(done / total)
    return h.hexdigest()


_DUR_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_VID_RE = re.compile(r"Stream #\d+:\d+.*?: Video: (\w+).*?,\s*(\d{2,5})x(\d{2,5}).*?(?:,\s*([\d.]+) fps)?")
_AUD_RE = re.compile(r"Stream #\d+:\d+.*?: Audio: (\w+).*?,\s*(\d+) Hz,\s*([^,]+)")


def probe(path: str | Path, compute_hash: bool = False) -> MediaInfo:
    """Probe a media file using ffmpeg's stderr banner plus OpenCV for exact
    frame counts. Works for both video and audio-only files."""
    path = str(path)
    info = MediaInfo(path=path)
    res = run_ffmpeg(["-i", path], text=True, encoding="utf-8", errors="replace")
    out = res.stderr or ""
    m = _DUR_RE.search(out)
    if m:
        info.duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    for line in out.splitlines():
        if "Video:" in line and "Stream #" in line and "attached pic" not in line:
            vm = _VID_RE.search(line)
            if vm and not info.has_video:
                info.has_video = True
                info.video_codec = vm.group(1)
                info.width, info.height = int(vm.group(2)), int(vm.group(3))
                fm = re.search(r"([\d.]+) fps", line)
                if fm:
                    info.fps = float(fm.group(1))
        elif "Audio:" in line and "Stream #" in line:
            am = _AUD_RE.search(line)
            if am and not info.has_audio:
                info.has_audio = True
                info.audio_codec = am.group(1)
                info.sample_rate = int(am.group(2))
                layout = am.group(3).strip()
                info.channels = {"mono": 1, "stereo": 2}.get(layout, 2 if "stereo" in layout else 1)
                cm = re.match(r"(\d+) channels", layout)
                if cm:
                    info.channels = int(cm.group(1))
    if info.has_video:
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 0:
                info.fps = float(fps)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w and h:
                info.width, info.height = w, h
            if n > 0:
                info.frame_count = n
            cap.release()
        if not info.frame_count and info.fps:
            info.frame_count = int(round(info.duration * info.fps))
        if not info.duration and info.fps and info.frame_count:
            info.duration = info.frame_count / info.fps
    if compute_hash:
        info.sha256 = sha256_file(path)
    return info


def decode_audio(path: str | Path, sample_rate: Optional[int] = None, mono: bool = False,
                 start: Optional[float] = None, duration: Optional[float] = None) -> tuple[np.ndarray, int]:
    """Decode the first audio stream to float32 PCM. Returns (samples[n, ch], sr)."""
    info = probe(path)
    if not info.has_audio:
        return np.zeros((0, 1), dtype=np.float32), sample_rate or 48000
    sr = sample_rate or info.sample_rate or 48000
    ch = 1 if mono else (info.channels or 1)
    args = []
    if start is not None:
        args += ["-ss", f"{start:.3f}"]
    args += ["-i", str(path)]
    if duration is not None:
        args += ["-t", f"{duration:.3f}"]
    args += ["-vn", "-map", "0:a:0", "-f", "f32le", "-acodec", "pcm_f32le", "-ac", str(ch), "-ar", str(sr), "pipe:1"]
    res = run_ffmpeg(args)
    if res.returncode != 0:
        raise RuntimeError("ffmpeg audio decode failed: " + res.stderr.decode("utf-8", "replace")[-800:])
    data = np.frombuffer(res.stdout, dtype=np.float32)
    data = data[: (len(data) // ch) * ch].reshape(-1, ch)
    return data.copy(), sr


def write_wav(path: str | Path, samples: np.ndarray, sr: int) -> None:
    import soundfile as sf
    sf.write(str(path), samples, sr, subtype="PCM_16")


class FrameReader:
    """Sequential/seekable BGR frame reader built on OpenCV."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._pos = 0

    @property
    def pos(self) -> int:
        """Index of the next frame that read() will return."""
        return self._pos

    def seek(self, frame: int) -> None:
        frame = max(0, frame)
        if frame == self._pos:
            return
        # Short forward hops are cheaper done by grabbing than by seeking
        # (seeking lands on a keyframe then decodes forward anyway).
        if 0 < frame - self._pos <= 30:
            for _ in range(frame - self._pos):
                if not self.cap.grab():
                    break
            self._pos = frame
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        self._pos = frame

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        if not ok:
            return None
        self._pos += 1
        return frame

    def read_at(self, frame: int) -> Optional[np.ndarray]:
        self.seek(frame)
        return self.read()

    def iter_frames(self, start: int = 0, end: Optional[int] = None) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (index, frame) for start..end-1."""
        self.seek(start)
        idx = start
        while end is None or idx < end:
            f = self.read()
            if f is None:
                break
            yield idx, f
            idx += 1

    def release(self) -> None:
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.release()
