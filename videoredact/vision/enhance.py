"""Detector-side brightness and contrast boost.

Body-cam footage is often dark, back-lit or washed out, and the detectors miss faces and
plates that a person can still make out. With this option on, every frame handed to a
detector is first enhanced: the luminance is stretched so that the darkest and brightest
half-percent of pixels sit at 0 and 255, then equalised locally (CLAHE, clip 2.0, 8x8 tiles)
so a face in a dark corner and one against a bright window both get contrast. Colour is left
alone (the L channel of Lab only).

Only the detector sees this copy. The preview, the frame the reviewer looks at and the
exported file are always built from the original pixels; the enhancement contributes boxes,
nothing else. The audit manifest records whether the option was on (`detect_enhance`).

Which detectors get it (ENHANCED_LABELS): the face detector and the plate detector. Measured
on our clips (bench/results/ENHANCE.md, bench/eval_enhance.py) the boost found faces and
plates the plain pass missed, but made the object detector lose the in-car laptop screen in
half the frames it had found it in, so YOLOX (screens, phones, documents, people) keeps the
original pixels. The export verification also runs on the original: a boosted pass there
would report the amplified texture of a steering-wheel emblem as an uncovered face.
"""
from __future__ import annotations

import cv2
import numpy as np

CLIP_LIMIT = 2.0        # CLAHE: how far a tile's histogram may be amplified (noise guard)
GRID = 8                # CLAHE tiles per side
STRETCH_LO = 0.005      # percentile mapped to 0 ...
STRETCH_HI = 0.995      # ... and to 255 before the local equalisation
MIN_RANGE = 8           # skip the stretch on a frame that is (nearly) one flat colour
NEAR_FULL = 3           # ... or one that already spans the range to within this many levels

ENHANCED_LABELS = frozenset({"face", "license_plate"})   # detectors that are handed the boosted copy

# Modes (settings "detect_enhance", AutoDetectOptions.enhance, TrackOptions.enhance):
#   off     the detectors see the original pixels
#   auto    only frames (or regions) that are dark or flat are boosted: mean grey below DARK_LUMA
#           or grey spread below FLAT_CONTRAST. On daylight footage this is a no-op, which is the
#           point: measured on our daytime clips (bench/results/ENHANCE.md) boosting every frame
#           found as many real faces as it lost and added texture false hits (a steering-wheel
#           emblem, the camera's timestamp overlay), while the few dark frames gained 2:1.
#   always  every frame is boosted (the experiment; for a clip that is dark throughout)
MODE_OFF, MODE_AUTO, MODE_ALWAYS = "off", "auto", "always"
MODES = (MODE_OFF, MODE_AUTO, MODE_ALWAYS)
DARK_LUMA = 60.0        # mean grey level (0..255) below which a frame counts as dark
FLAT_CONTRAST = 30.0    # grey standard deviation below which a frame counts as washed out / flat
STATS_SIDE = 160        # the statistics are taken on a thumbnail this wide (sub-millisecond)


def as_mode(v) -> str:
    """Normalise a settings value: True -> always, False/None/"" -> off, else one of MODES."""
    if v is True:
        return MODE_ALWAYS
    if not v:
        return MODE_OFF
    s = str(v).strip().lower()
    return s if s in MODES else MODE_AUTO


def needs_boost(frame: np.ndarray) -> bool:
    """Dark or flat by the thresholds above (measured on a thumbnail)."""
    if frame is None or frame.ndim != 3:
        return False
    step = max(1, frame.shape[1] // STATS_SIDE)
    m, s = luma_stats(np.ascontiguousarray(frame[::step, ::step]))   # subsampled, not averaged: keeps the pixel spread
    return m < DARK_LUMA or s < FLAT_CONTRAST


def boost(frame: np.ndarray, mode: str) -> np.ndarray:
    """What a detector is shown under `mode` (see MODES): the frame itself, or its enhanced copy."""
    mode = as_mode(mode)
    if mode == MODE_OFF or (mode == MODE_AUTO and not needs_boost(frame)):
        return frame
    return enhance_for_detection(frame)

_clahe: dict[tuple[float, int], object] = {}


def _get_clahe(clip: float, grid: int):
    key = (float(clip), int(grid))
    if key not in _clahe:
        _clahe[key] = cv2.createCLAHE(clipLimit=key[0], tileGridSize=(key[1], key[1]))
    return _clahe[key]


def luma_stats(frame: np.ndarray) -> tuple[float, float]:
    """(mean, standard deviation) of the grey level, 0..255: how dark and how flat a frame is."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    m, s = cv2.meanStdDev(g)
    return float(m[0][0]), float(s[0][0])


def stretch_lut(L: np.ndarray) -> np.ndarray | None:
    """Lookup table mapping the STRETCH_LO..STRETCH_HI percentiles of `L` to 0..255, or None
    when the frame already spans the range (or is flat)."""
    hist = cv2.calcHist([L], [0], None, [256], [0, 256]).ravel()
    total = float(hist.sum())
    if total <= 0:
        return None
    cdf = np.cumsum(hist) / total
    lo = int(np.searchsorted(cdf, STRETCH_LO))
    hi = int(np.searchsorted(cdf, STRETCH_HI))
    if hi - lo < MIN_RANGE or (lo <= NEAR_FULL and hi >= 255 - NEAR_FULL):
        return None
    return np.clip((np.arange(256, dtype=np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def enhance_for_detection(frame: np.ndarray, clip: float = None, grid: int = None) -> np.ndarray:
    """A new BGR frame for the detectors: luminance stretched and locally equalised. The input
    is not modified. `clip` / `grid` override the module defaults (bench variants); clip 0
    skips the local equalisation and keeps only the stretch."""
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        return frame
    clip = CLIP_LIMIT if clip is None else float(clip)
    grid = GRID if grid is None else int(grid)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    lut = stretch_lut(L)
    if lut is not None:
        L = cv2.LUT(L, lut)
    lab[:, :, 0] = _get_clahe(clip, grid).apply(L) if clip > 0 else L
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
