"""Self-contained preview player (no QMediaPlayer).

Why: QMediaPlayer stalled with audio output on the development machine with
both the FFmpeg and Windows backends whenever a window was visible. This
player only depends on things we already rely on for export:

  * video frames  - OpenCV FrameReader (sequential decode, frame-exact stepping)
  * audio         - FFmpeg subprocess streaming s16le PCM from any position,
                    pushed into a QAudioSink; QAudioSink.processedUSecs() is
                    the master clock while audio is present.
  * preview       - audio redactions (beep/silence/...) are applied to each
                    PCM chunk exactly as the export does, so the preview is a
                    faithful rendition of the output.

Public API (mirrors the subset of QMediaPlayer the UI uses):
  load(info, project) · play() · pause() · toggle() · stop() · seek(seconds)
  step(n_frames) · position_s() · current_frame() · is_playing
Signals: positionChanged(ms), stateChanged(playing), frameReady(bgr, idx),
         error(str), ended()
"""
from __future__ import annotations

import queue
import subprocess
import sys
import threading
import time
from typing import Optional

import numpy as np
from PySide6.QtCore import QIODevice, QObject, QTimer, Signal
from PySide6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices

from videoredact.core.audio_redact import apply_redactions
from videoredact.core.media import FrameReader, ffmpeg_exe
from videoredact.core.model import MediaInfo, Project

_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
CHUNK_FRAMES = 2048          # PCM frames per chunk pushed to the sink (~43 ms @ 48 kHz)
QUEUE_CHUNKS = 12
TICK_MS = 8


class _PcmStream:
    """FFmpeg -> s16le PCM pipe read by a daemon thread into a queue."""

    def __init__(self, path: str, start_s: float, sr: int, ch: int):
        self.sr, self.ch = sr, ch
        self.bytes_per_frame = 2 * ch
        cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-ss", f"{max(0.0, start_s):.3f}",
               "-i", path, "-vn", "-map", "0:a:0", "-f", "s16le", "-acodec", "pcm_s16le",
               "-ac", str(ch), "-ar", str(sr), "pipe:1"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     creationflags=_CREATE_NO_WINDOW)
        self.q: "queue.Queue[bytes]" = queue.Queue(maxsize=QUEUE_CHUNKS)
        self.eof = False
        self._stop = False
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self) -> None:
        n = CHUNK_FRAMES * self.bytes_per_frame
        try:
            while not self._stop:
                data = self.proc.stdout.read(n)
                if not data:
                    break
                while not self._stop:
                    try:
                        self.q.put(data, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        finally:
            self.eof = True

    def get(self) -> Optional[bytes]:
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        self._stop = True
        try:
            self.proc.kill()
        except Exception:
            pass
        try:
            self.proc.stdout.close()
        except Exception:
            pass


class MediaPlayer(QObject):
    positionChanged = Signal(int)
    stateChanged = Signal(bool)
    frameReady = Signal(object, int)
    error = Signal(str)
    ended = Signal()

    def __init__(self):
        super().__init__()
        self.info: Optional[MediaInfo] = None
        self.project: Optional[Project] = None
        self.reader: Optional[FrameReader] = None
        self.preview_redactions = True
        self.volume = 0.8
        self._playing = False
        self._pos_s = 0.0            # authoritative position while paused
        self._base_s = 0.0           # position at which the current play run started
        self._wall0 = 0.0
        self._sink: Optional[QAudioSink] = None
        self._dev: Optional[QIODevice] = None
        self._stream: Optional[_PcmStream] = None
        self._chunk_t = 0.0          # time of the next PCM chunk to push
        self._shown_idx = -1
        self._sr = 48000
        self._ch = 2
        self._timer = QTimer(self)
        self._timer.setInterval(TICK_MS)
        self._timer.timeout.connect(self._tick)
        self._pos_timer = QTimer(self)
        self._pos_timer.setInterval(40)
        self._pos_timer.timeout.connect(lambda: self.positionChanged.emit(int(self.position_s() * 1000)))

    # ------------------------------------------------------------- loading
    def load(self, info: MediaInfo, project: Optional[Project]) -> None:
        self.stop()
        if self.reader:
            self.reader.release()
            self.reader = None
        self.info = info
        self.project = project
        self._pos_s = 0.0
        self._shown_idx = -1
        if info.has_video:
            try:
                self.reader = FrameReader(info.path)
            except Exception as e:  # noqa: BLE001
                self.error.emit(str(e))
        if info.has_audio:
            self._sr = info.sample_rate or 48000
            self._ch = min(2, max(1, info.channels or 2))
        self._show_frame_at(0)
        self.positionChanged.emit(0)

    @property
    def fps(self) -> float:
        return self.info.fps if self.info and self.info.fps else 30.0

    @property
    def duration(self) -> float:
        return self.info.duration if self.info else 0.0

    @property
    def is_playing(self) -> bool:
        return self._playing

    # ------------------------------------------------------------ position
    def position_s(self) -> float:
        if not self._playing:
            return self._pos_s
        if self._sink is not None:
            return self._base_s + self._sink.processedUSecs() / 1e6
        return self._base_s + (time.perf_counter() - self._wall0)

    def position(self) -> int:
        return int(self.position_s() * 1000)

    def current_frame(self) -> int:
        return int(self.position_s() * self.fps + 1e-6)

    # ------------------------------------------------------------ control
    def play(self) -> None:
        if not self.info or self._playing:
            return
        if self._pos_s >= self.duration - 0.02 and self.duration > 0:
            self._pos_s = 0.0
            self._show_frame_at(0)
        self._base_s = self._pos_s
        self._wall0 = time.perf_counter()
        if self.info.has_audio:
            self._start_audio(self._base_s)
        self._playing = True
        self._timer.start()
        self._pos_timer.start()
        self.stateChanged.emit(True)

    def pause(self) -> None:
        if not self._playing:
            return
        self._pos_s = min(self.position_s(), self.duration)
        self._playing = False
        self._timer.stop()
        self._pos_timer.stop()
        self._stop_audio()
        self.stateChanged.emit(False)
        self.positionChanged.emit(int(self._pos_s * 1000))

    def toggle(self) -> None:
        if self._playing:
            self.pause()
        else:
            self.play()

    def stop(self) -> None:
        if self._playing:
            self.pause()
        else:
            self._stop_audio()

    def seek(self, t: float) -> None:
        if not self.info:
            return
        t = min(max(0.0, t), max(0.0, self.duration))
        was = self._playing
        if was:
            self.pause()
        self._pos_s = t
        self._show_frame_at(int(t * self.fps + 1e-6))
        self.positionChanged.emit(int(t * 1000))
        if was:
            self.play()

    def step(self, n: int) -> None:
        if not self.info:
            return
        if self._playing:
            self.pause()
        idx = max(0, self.current_frame() + n)
        if self.info.frame_count:
            idx = min(idx, self.info.frame_count - 1)
        self._pos_s = idx / self.fps + 0.5 / self.fps  # mid-frame: robust against rounding
        if self.reader and n == 1 and self.reader.pos == idx:
            f = self.reader.read()
            if f is not None:
                self._shown_idx = idx
                self.frameReady.emit(f, idx)
        else:
            self._show_frame_at(idx)
        self.positionChanged.emit(int(self._pos_s * 1000))

    def set_volume(self, v: float) -> None:
        self.volume = max(0.0, min(1.0, v))
        if self._sink is not None:
            self._sink.setVolume(self.volume)

    # ------------------------------------------------------------- frames
    def _show_frame_at(self, idx: int) -> None:
        if not self.reader:
            return
        if self.info.frame_count:
            idx = min(idx, max(0, self.info.frame_count - 1))
        f = self.reader.read_at(idx)
        if f is None and idx > 0:  # past the end: show the last readable frame
            f = self.reader.read_at(idx - 1)
            idx -= 1
        if f is not None:
            self._shown_idx = idx
            self.frameReady.emit(f, idx)

    def _tick(self) -> None:
        if not self._playing:
            return
        self._feed_audio()
        pos = self.position_s()
        if self.duration and pos >= self.duration:
            self.pause()
            self._pos_s = self.duration
            self.positionChanged.emit(int(self._pos_s * 1000))
            self.ended.emit()
            return
        if self.reader:
            target = int(pos * self.fps + 1e-6)
            if target > self._shown_idx:
                if target - self.reader.pos > 3:      # fell behind: skip ahead
                    self.reader.seek(target)
                elif self.reader.pos < target:
                    while self.reader.pos < target:
                        if not self.reader.cap.grab():
                            break
                        self.reader._pos += 1
                f = self.reader.read()
                if f is not None:
                    self._shown_idx = self.reader.pos - 1
                    self.frameReady.emit(f, self._shown_idx)

    # -------------------------------------------------------------- audio
    def _start_audio(self, t: float) -> None:
        self._stop_audio()
        fmt = QAudioFormat()
        fmt.setSampleRate(self._sr)
        fmt.setChannelCount(self._ch)
        fmt.setSampleFormat(QAudioFormat.Int16)
        dev = QMediaDevices.defaultAudioOutput()
        if dev.isNull():
            self._sink = None
            return
        if not dev.isFormatSupported(fmt):
            fmt = dev.preferredFormat()
            fmt.setSampleFormat(QAudioFormat.Int16)
            self._sr, self._ch = fmt.sampleRate(), fmt.channelCount()
        try:
            self._stream = _PcmStream(self.info.path, t, self._sr, self._ch)
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"Audio preview unavailable: {e}")
            self._stream = None
            return
        self._sink = QAudioSink(dev, fmt)
        self._sink.setBufferSize(int(self._sr * self._ch * 2 * 0.15))  # ~150 ms
        self._sink.setVolume(self.volume)
        self._dev = self._sink.start()
        self._chunk_t = t
        self._feed_audio()

    def _stop_audio(self) -> None:
        if self._sink is not None:
            try:
                self._sink.stop()
            except Exception:
                pass
            self._sink = None
            self._dev = None
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def _feed_audio(self) -> None:
        if self._sink is None or self._dev is None or self._stream is None:
            return
        bpf = 2 * self._ch
        while self._sink.bytesFree() >= CHUNK_FRAMES * bpf:
            data = self._stream.get()
            if data is None:
                break
            n = len(data) // bpf
            if self.preview_redactions and self.project and self.project.audio_redactions:
                x = np.frombuffer(data, dtype=np.int16).reshape(-1, self._ch).astype(np.float32) / 32768.0
                x = apply_redactions(x, self._sr, self.project.audio_redactions, self.project.default_audio_style,
                                     self.project.beep_frequency, start_time=self._chunk_t)
                data = np.clip(x * 32767.0, -32768, 32767).astype(np.int16).tobytes()
            self._dev.write(data)
            self._chunk_t += n / self._sr
