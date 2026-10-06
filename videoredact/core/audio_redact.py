"""Apply audio redactions (beep / silence / low tone / noise) to PCM samples.

The requested range is overwritten completely: nothing of the redacted
speech survives inside [start, end). Short raised-cosine crossfades are
applied just *outside* the range so the result has no clicks.
"""
from __future__ import annotations

import numpy as np

from .model import AudioRedaction, AudioStyle, as_enum

LOW_TONE_FREQ = 400.0
DEFAULT_LEVEL_DB = -14.0


def merge_ranges(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping/adjacent (start, end) ranges."""
    out: list[tuple[float, float]] = []
    for s, e in sorted(ranges):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def _fill(style: AudioStyle, n: int, sr: int, channels: int, beep_freq: float,
          level_db: float, rng: np.random.Generator, offset: int = 0) -> np.ndarray:
    amp = 10 ** (level_db / 20)
    if style == AudioStyle.SILENCE:
        return np.zeros((n, channels), dtype=np.float32)
    if style == AudioStyle.NOISE:
        return (rng.standard_normal((n, channels)) * amp * 0.3).astype(np.float32)
    freq = beep_freq if style == AudioStyle.BEEP else LOW_TONE_FREQ
    t = np.arange(offset, offset + n, dtype=np.float64) / sr
    tone = (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)
    return np.repeat(tone[:, None], channels, axis=1)


def apply_redactions(samples: np.ndarray, sr: int, redactions: list[AudioRedaction],
                     default_style: AudioStyle = AudioStyle.BEEP, beep_freq: float = 1000.0,
                     fade_ms: float = 8.0, level_db: float = DEFAULT_LEVEL_DB,
                     pad_s: float = 0.0, start_time: float = 0.0) -> np.ndarray:
    """Return a new array with every enabled redaction applied.

    samples: float32 [n, channels] (or [n]); returned in the same shape.
    pad_s: extra seconds added on both sides of each redaction (helps cover
           word-boundary timing error from the transcriber).
    start_time: absolute time (s) of samples[0]; lets a streaming preview
           process the file in chunks with phase-continuous tones.
    """
    squeeze = samples.ndim == 1
    x = samples.reshape(-1, 1) if squeeze else samples
    out = x.astype(np.float32, copy=True)
    n_total, ch = out.shape
    fade = int(sr * fade_ms / 1000)
    rng = np.random.default_rng(12345)
    base = int(round(start_time * sr))

    by_style: dict[AudioStyle, list[tuple[float, float]]] = {}
    for r in redactions:
        if not r.enabled or r.end <= r.start:
            continue
        st = as_enum(AudioStyle, r.style or default_style)
        by_style.setdefault(st, []).append((max(0.0, r.start - pad_s) - start_time, r.end + pad_s - start_time))

    for style, ranges in by_style.items():
        for s, e in merge_ranges(ranges):
            i0 = max(0, int(round(s * sr)))
            i1 = min(n_total, int(round(e * sr)))
            if i1 <= i0:
                continue
            # Fades live outside the hard range so the interior is 100% replacement.
            a0 = max(0, i0 - fade)
            a1 = min(n_total, i1 + fade)
            n = a1 - a0
            repl = _fill(style, n, sr, ch, beep_freq, level_db, rng, offset=base + a0)
            fin = i0 - a0
            fout = a1 - i1
            if fin > 0:
                ramp = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fin, endpoint=False)))[:, None].astype(np.float32)
                repl[:fin] = out[a0:i0] * (1 - ramp) + repl[:fin] * ramp
            if fout > 0:
                ramp = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fout, endpoint=False)))[:, None].astype(np.float32)
                repl[n - fout:] = out[i1:a1] * ramp + repl[n - fout:] * (1 - ramp)
            out[a0:a1] = repl
    return out.reshape(-1) if squeeze else out
