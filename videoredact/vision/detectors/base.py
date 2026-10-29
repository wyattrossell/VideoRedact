"""Pluggable detector interface.

    class Detector:
        labels: set[str]                       # labels this detector can emit
        def detect(self, frames) -> list[list[Detection]]   # batch API
        def detect_one(self, frame) -> list[Detection]

Every model (YuNet, SCRFD, CenterFace, YOLOX, EgoBlur, open-vocabulary) is an
adapter implementing `detect`. Wrappers (`TiledDetector`, `MultiScaleDetector`,
`EnsembleDetector`, `HeadFallback`) are themselves Detectors, so pipelines are
composed from config (see `build()` in `__init__.py`) and the rest of the app
only ever sees `Detector.detect`.
Boxes are normalized `BBox` (0..1) so wrappers can rescale freely.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from videoredact.core.model import BBox


@dataclass
class Detection:
    bbox: BBox
    label: str
    score: float
    source: str = ""      # which detector produced it (for ensembles / review flags)
    landmarks: Optional[np.ndarray] = None   # faces: 5 x (x, y) normalized to the frame, when the model gives them


class Detector(ABC):
    name: str = "detector"
    labels: set[str] = set()

    @abstractmethod
    def detect(self, frames: Sequence[np.ndarray]) -> list[list[Detection]]:
        """Run on a batch of BGR frames; returns one detection list per frame."""

    def detect_one(self, frame: np.ndarray) -> list[Detection]:
        return self.detect([frame])[0]

    def close(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.name} {sorted(self.labels)}>"


class PerFrameDetector(Detector):
    """Base for models without batch support: implement `_detect_frame`."""

    def detect(self, frames: Sequence[np.ndarray]) -> list[list[Detection]]:
        return [self._detect_frame(f) for f in frames]

    @abstractmethod
    def _detect_frame(self, frame: np.ndarray) -> list[Detection]:
        ...


def nms_merge(dets: Iterable[Detection], iou_thr: float = 0.5, class_aware: bool = True) -> list[Detection]:
    """Greedy NMS over detections (per label when class_aware). Keeps the highest score;
    the kept box is the union-weighted average of overlapping boxes (weighted box fusion
    lite) so tiles/scales that see the same object agree on one box."""
    dets = sorted(dets, key=lambda d: d.score, reverse=True)
    out: list[Detection] = []
    used = [False] * len(dets)
    for i, d in enumerate(dets):
        if used[i]:
            continue
        group = [d]
        used[i] = True
        for j in range(i + 1, len(dets)):
            if used[j]:
                continue
            e = dets[j]
            if class_aware and e.label != d.label:
                continue
            if d.bbox.iou(e.bbox) >= iou_thr or _contains(d.bbox, e.bbox, 0.85) or _contains(e.bbox, d.bbox, 0.85):
                group.append(e)
                used[j] = True
        if len(group) == 1:
            out.append(d)
            continue
        wsum = sum(g.score for g in group)
        x = sum(g.bbox.x * g.score for g in group) / wsum
        y = sum(g.bbox.y * g.score for g in group) / wsum
        w = sum(g.bbox.w * g.score for g in group) / wsum
        h = sum(g.bbox.h * g.score for g in group) / wsum
        src = "+".join(sorted({g.source for g in group if g.source}))
        out.append(Detection(BBox(x, y, w, h).clamp(), d.label, d.score, src))
    return out


def _contains(outer: BBox, inner: BBox, frac: float) -> bool:
    """True if at least `frac` of inner's area lies inside outer."""
    ix0, iy0 = max(outer.x, inner.x), max(outer.y, inner.y)
    ix1, iy1 = min(outer.x + outer.w, inner.x + inner.w), min(outer.y + outer.h, inner.y + inner.h)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area = inner.w * inner.h
    return area > 0 and inter / area >= frac
