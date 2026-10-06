"""Apply video redactions (black box / blur / pixelate, rect or ellipse) to frames.

Blur and pixelate strengths scale with the box size so a small face and a
large monitor are both unrecognizable. Pixelate uses a coarse block grid
(default ~8 blocks across the box); blur uses a kernel ~1/4 of the box size
applied twice. Both are irreversible at these strengths.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .model import BBox, Shape, VideoStyle, VideoTrack

PIXELATE_BLOCKS = 8     # blocks across the box's shorter side
BLUR_FRACTION = 0.25    # kernel size as fraction of the box's shorter side


def _region_styled(roi: np.ndarray, style: VideoStyle) -> np.ndarray:
    h, w = roi.shape[:2]
    if style == VideoStyle.BLACK:
        return np.zeros_like(roi)
    short = max(1, min(h, w))
    if style == VideoStyle.PIXELATE:
        bw = max(1, w * PIXELATE_BLOCKS // short)
        bh = max(1, h * PIXELATE_BLOCKS // short)
        small = cv2.resize(roi, (bw, bh), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    if style == VideoStyle.BLUR:
        k = max(3, int(short * BLUR_FRACTION) | 1)
        # downscale -> blur -> upscale is both faster and stronger than a huge kernel
        ds = max(1, short // 32)
        small = cv2.resize(roi, (max(1, w // ds), max(1, h // ds)), interpolation=cv2.INTER_AREA)
        ks = max(3, (k // ds) | 1)
        small = cv2.GaussianBlur(small, (ks, ks), 0)
        small = cv2.GaussianBlur(small, (ks, ks), 0)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    raise ValueError(style)


def apply_box(frame: np.ndarray, bbox: BBox, style: VideoStyle, shape: Shape = Shape.RECT,
              pad: float = 0.0) -> None:
    """Redact one box in-place."""
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = bbox.to_pixels(W, H, pad)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return
    roi = frame[y0:y1, x0:x1]
    styled = _region_styled(roi, style)
    if shape == Shape.ELLIPSE:
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.ellipse(mask, ((x1 - x0) // 2, (y1 - y0) // 2), ((x1 - x0) // 2, (y1 - y0) // 2), 0, 0, 360, 255, -1)
        roi[mask > 0] = styled[mask > 0]
    else:
        frame[y0:y1, x0:x1] = styled


def redact_frame(frame: np.ndarray, tracks: list[VideoTrack], frame_idx: int,
                 default_style: VideoStyle, in_place: bool = True) -> np.ndarray:
    """Apply every enabled track that has a box at frame_idx."""
    out = frame if in_place else frame.copy()
    for t in tracks:
        if not t.enabled:
            continue
        bb = t.bbox_at(frame_idx)
        if bb is not None:
            apply_box(out, bb, t.style or default_style, t.shape, t.pad)
    return out


def boxes_at(tracks: list[VideoTrack], frame_idx: int) -> list[tuple[VideoTrack, BBox]]:
    res = []
    for t in tracks:
        bb = t.bbox_at(frame_idx)
        if bb is not None:
            res.append((t, bb))
    return res
