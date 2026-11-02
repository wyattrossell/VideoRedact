"""Apply video redactions (black or blue box; rect, ellipse or rounded) to frames.

The program offers solid fills only (decision of 2026-11-02): a solid box leaves nothing to
recover. Blur and pixelate are kept here for the re-identification bench
(bench/reid_attack.py) and for the irreversibility tests; a project that used them is
converted to black when it is opened (core/model.py Project.retire_styles). Their strengths
scale with the box size: pixelate uses a coarse block grid (default ~8 blocks across the box,
never under 16 px), blur averages the region down to ~6 samples across.
"""
from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from .model import BBox, Shape, VideoStyle, VideoTrack, as_enum

PIXELATE_BLOCKS = 8     # blocks across the box's shorter side ...
PIXELATE_MIN_BLOCK = 16  # ... but never smaller than this many output pixels per block (irreversibility floor)
BLUR_FRACTION = 0.25    # kernel size as fraction of the box's shorter side
BLUR_DOWNSCALE = 6      # heavy blur: the region is reduced to at most this many pixels across before blurring
BLUE_BGR = (180, 70, 20)  # "blue box": a solid fill like black, in a colour that reads as a deliberate redaction


def _region_styled(roi: np.ndarray, style: VideoStyle) -> np.ndarray:
    h, w = roi.shape[:2]
    if style == VideoStyle.BLACK:
        return np.zeros_like(roi)
    if style == VideoStyle.BLUE:
        out = np.empty_like(roi)
        out[:] = BLUE_BGR
        return out
    short = max(1, min(h, w))
    if style == VideoStyle.PIXELATE:
        # block size = short side / PIXELATE_BLOCKS, but at least PIXELATE_MIN_BLOCK px: a 30 px
        # face becomes 2x2 blocks, a 600 px screen 8 blocks across. Each block is the mean colour
        # of its area; nothing finer survives (de-pixelation attacks need many more blocks).
        block = max(PIXELATE_MIN_BLOCK, short // PIXELATE_BLOCKS)
        bw = max(1, int(round(w / block)))
        bh = max(1, int(round(h / block)))
        small = cv2.resize(roi, (bw, bh), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    if style == VideoStyle.BLUR:
        # "heavy blur": average the region down to <= BLUR_DOWNSCALE px across, blur that, scale back.
        # This is a smooth-looking pixelation at ~6 blocks: light Gaussian blur was deliberately
        # removed because it is partially reversible (see docs/CJIS_COMPLIANCE.md).
        ds = max(1, short // BLUR_DOWNSCALE)
        small = cv2.resize(roi, (max(1, w // ds), max(1, h // ds)), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (3, 3), 0)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    raise ValueError(style)


def apply_box(frame: np.ndarray, bbox: BBox, style: VideoStyle, shape: Shape = Shape.RECT,
              pad: float = 0.0) -> None:
    """Redact one box in-place."""
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = bbox.to_pixels(W, H, pad)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return
    style = as_enum(VideoStyle, style)
    shape = as_enum(Shape, shape)
    roi = frame[y0:y1, x0:x1]
    styled = _region_styled(roi, style)
    if shape == Shape.ELLIPSE:
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.ellipse(mask, ((x1 - x0) // 2, (y1 - y0) // 2), ((x1 - x0) // 2, (y1 - y0) // 2), 0, 0, 360, 255, -1)
        cv2.copyTo(styled, mask, roi)          # masked copy in C++ (numpy boolean indexing cost ~10x more)
    elif shape == Shape.ROUNDED:
        h, w = roi.shape[:2]
        r = max(2, int(min(w, h) * 0.25))
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (r, 0), (w - r, h), 255, -1)
        cv2.rectangle(mask, (0, r), (w, h - r), 255, -1)
        for cx, cy in ((r, r), (w - r, r), (r, h - r), (w - r, h - r)):
            cv2.circle(mask, (cx, cy), r, 255, -1)
        cv2.copyTo(styled, mask, roi)
    else:
        frame[y0:y1, x0:x1] = styled


def _save_zones(frame: np.ndarray, zones: Sequence[BBox]) -> list[tuple[int, int, int, int, np.ndarray]]:
    H, W = frame.shape[:2]
    saved = []
    for z in zones:
        x0, y0, x1, y1 = z.to_pixels(W, H)
        if x1 > x0 and y1 > y0:
            saved.append((y0, y1, x0, x1, frame[y0:y1, x0:x1].copy()))
    return saved


def _restore_zones(frame: np.ndarray, saved) -> None:
    for y0, y1, x0, x1, px in saved:
        frame[y0:y1, x0:x1] = px


def redact_frame(frame: np.ndarray, tracks: list[VideoTrack], frame_idx: int,
                 default_style: VideoStyle, in_place: bool = True,
                 protected: Optional[Sequence[BBox]] = None) -> np.ndarray:
    """Apply every enabled track that has a box at frame_idx. `protected` zones (the camera's
    timestamp overlay, core/model.py ProtectedZone) get their original pixels back afterwards,
    so no box ever covers them."""
    out = frame if in_place else frame.copy()
    saved = _save_zones(out, protected) if protected else None
    for t in tracks:
        if not t.enabled:
            continue
        bb = t.bbox_at(frame_idx)
        if bb is not None:
            apply_box(out, bb, t.style or default_style, t.shape, t.pad)
    if saved:
        _restore_zones(out, saved)
    return out


def boxes_at(tracks: list[VideoTrack], frame_idx: int) -> list[tuple[VideoTrack, BBox]]:
    res = []
    for t in tracks:
        bb = t.bbox_at(frame_idx)
        if bb is not None:
            res.append((t, bb))
    return res
