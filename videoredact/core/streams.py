"""Producer/consumer building blocks for the video pipelines.

FrameProducer   decodes frames on its own thread into a bounded queue. Source is
                either OpenCV VideoCapture (fastest full-resolution path, measured
                764 fps @720p / 264 fps @1080p) or an FFmpeg raw-video pipe with a
                scale filter that yields a downscaled *proxy* directly (QuickSync or
                CPU, ~480-500 fps regardless of source resolution; full-resolution
                pipes are bandwidth-bound and slower than OpenCV). `auto` picks the
                pipe whenever a proxy smaller than the source is wanted.
FFmpegWriter    feeds raw BGR frames to an FFmpeg encoder on its own thread from a
                bounded queue, so decode -> redact -> encode overlap instead of
                running serially.

Both release the GIL in their hot paths (OpenCV / pipe I/O), so Python threads
are enough; no extra processes.
"""
from __future__ import annotations

import queue
import subprocess
import sys
import threading
from typing import Iterator, Optional

import numpy as np

from .media import FrameReader, ffmpeg_exe

_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
_SENTINEL = object()


def _hwaccel_args(hw: Optional[str]) -> list[str]:
    """Decoder flags. Hardware decoders keep frames on the device, so the scale must run
    there (scale_qsv / scale_cuda) followed by hwdownload; d3d11va downloads first."""
    if not hw or hw == "none":
        return []
    out = ["-hwaccel", hw]
    if hw in ("qsv", "cuda"):
        out += ["-hwaccel_output_format", hw]
    return out


def _scale_filter(hw: Optional[str], w: int, h: int) -> str:
    if hw == "qsv":
        return f"scale_qsv=w={w}:h={h},hwdownload,format=nv12"
    if hw == "cuda":
        return f"scale_cuda={w}:{h},hwdownload,format=nv12"
    if hw == "d3d11va":
        return f"scale={w}:{h}:flags=fast_bilinear"   # d3d11va frames are already in system memory
    return f"scale={w}:{h}:flags=fast_bilinear"


class FrameProducer:
    """Iterate `(index, frame)` for frames start..end-1 decoded on a background thread.

    proxy_width: if set and smaller than the source width, frames come from an FFmpeg
    pipe scaled to that width (height keeps the aspect, even). hwaccel: None/'auto'/
    'qsv'/'d3d11va'/'cuda' for the pipe path ('auto' tries qsv then CPU).
    """

    def __init__(self, path: str, start: int = 0, end: Optional[int] = None, proxy_width: Optional[int] = None,
                 hwaccel: Optional[str] = "auto", queue_size: int = 32, step: int = 1):
        self.path, self.start, self.step = path, start, max(1, step)
        info = FrameReader(path)
        self.src_w, self.src_h, self.fps, self.frame_count = info.width, info.height, info.fps, info.frame_count
        info.release()
        self.end = end if end is not None else self.frame_count
        self.proxy = bool(proxy_width and proxy_width < self.src_w)
        if self.proxy:
            self.w = int(proxy_width)
            self.h = int(round(self.src_h * self.w / self.src_w / 2) * 2)
        else:
            self.w, self.h = self.src_w, self.src_h
        self.hwaccel = hwaccel
        self.q: "queue.Queue" = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self.error: Optional[str] = None
        self.engine = "ffmpeg-pipe" if self.proxy else "opencv"
        self._t = threading.Thread(target=self._run_pipe if self.proxy else self._run_cv2, daemon=True)
        self._t.start()

    # ---- producers --------------------------------------------------------
    def _run_cv2(self) -> None:
        try:
            with FrameReader(self.path) as r:
                r.seek(self.start)
                idx = self.start
                while idx < self.end and not self._stop.is_set():
                    if self.step > 1 and (idx - self.start) % self.step:
                        if not r.cap.grab():
                            break
                        r._pos += 1
                        idx += 1
                        continue
                    f = r.read()
                    if f is None:
                        break
                    self._put((idx, f))
                    idx += 1
        except Exception as e:  # noqa: BLE001
            self.error = str(e)
        finally:
            self._put(_SENTINEL)

    def _pipe_cmd(self, hw: Optional[str]) -> list[str]:
        cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin"]
        cmd += _hwaccel_args(hw)
        if self.start > 0:
            cmd += ["-ss", f"{self.start / self.fps:.6f}"]
        cmd += ["-i", self.path, "-an"]
        n = self.end - self.start
        if n > 0:
            cmd += ["-frames:v", str(n)]
        vf = []
        if self.step > 1:
            vf.append(f"select='not(mod(n\\,{self.step}))'")
        vf.append(_scale_filter(hw, self.w, self.h))
        if self.step > 1:
            vf.append("setpts=N/FRAME_RATE/TB")
        cmd += ["-vf", ",".join(vf), "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
        return cmd

    def _run_pipe(self) -> None:
        fsz = self.w * self.h * 3
        # None = software decode only; "auto" = try QuickSync, then software; else the named decoder
        hw_order = ["qsv", None] if self.hwaccel == "auto" else [None if self.hwaccel in (None, "none") else self.hwaccel]
        for hw in hw_order:
            if self._stop.is_set():
                break
            try:
                p = subprocess.Popen(self._pipe_cmd(hw), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     bufsize=fsz * 4, creationflags=_CREATE_NO_WINDOW)
            except Exception as e:  # noqa: BLE001
                self.error = str(e)
                continue
            idx = self.start
            got = 0
            while not self._stop.is_set():
                buf = p.stdout.read(fsz)
                if len(buf) < fsz:
                    break
                self._put((idx, np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3).copy()))
                idx += self.step
                got += 1
            try:
                p.stdout.close()
            except Exception:
                pass
            err = p.stderr.read().decode("utf-8", "replace")
            p.wait()
            if got > 0 or self._stop.is_set():
                self.engine = f"ffmpeg-pipe/{hw or 'cpu'}"
                self.error = None
                break
            self.error = err[-300:] or f"ffmpeg produced no frames with hwaccel={hw}"
            # try the next decoder (e.g. qsv missing -> cpu)
        self._put(_SENTINEL)

    def _put(self, item) -> None:
        while not self._stop.is_set():
            try:
                self.q.put(item, timeout=0.2)
                return
            except queue.Full:
                continue

    # ---- consumer -----------------------------------------------------------
    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        while True:
            item = self.q.get()
            if item is _SENTINEL:
                if self.error:
                    raise RuntimeError(self.error)
                return
            yield item

    def close(self) -> None:
        self._stop.set()
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
        self._t.join(timeout=2)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class FFmpegWriter:
    """Background writer: `put(frame)` queues a BGR frame, a thread streams it into
    FFmpeg's stdin. `close()` flushes, waits, and raises if FFmpeg failed."""

    def __init__(self, cmd: list[str], queue_size: int = 48):
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_CREATE_NO_WINDOW)
        self.q: "queue.Queue" = queue.Queue(maxsize=queue_size)
        self.error: Optional[str] = None
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self) -> None:
        try:
            while True:
                item = self.q.get()
                if item is _SENTINEL:
                    break
                self.proc.stdin.write(item)
        except Exception as e:  # noqa: BLE001
            self.error = str(e)
            try:
                while self.q.get_nowait() is not _SENTINEL:
                    pass
            except queue.Empty:
                pass
        finally:
            try:
                self.proc.stdin.close()
            except Exception:
                pass

    def put(self, frame: np.ndarray) -> None:
        if self.error:
            raise RuntimeError(f"FFmpeg writer failed: {self.error}")
        self.q.put(frame.tobytes())

    def abort(self) -> None:
        try:
            self.proc.kill()
        except Exception:
            pass
        self.q.put(_SENTINEL)
        self._t.join(timeout=2)

    def close(self) -> str:
        """Finish encoding; returns FFmpeg's stderr; raises on non-zero exit."""
        self.q.put(_SENTINEL)
        self._t.join()
        err = self.proc.stderr.read().decode("utf-8", "replace")
        rc = self.proc.wait()
        if rc != 0:
            raise RuntimeError(f"FFmpeg encode failed (code {rc}): {err[-1500:]}")
        return err
