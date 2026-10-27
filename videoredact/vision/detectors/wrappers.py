"""Detector combinators: tiling (SAHI-style), multi-scale, ensembles, head fallback."""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from videoredact.core.model import BBox
from .base import Detection, Detector, nms_merge, _contains


class TiledDetector(Detector):
    """Slice the frame into overlapping tiles so small/distant objects are seen at
    full resolution (SAHI idea), run the inner detector on the full frame plus every
    tile, and fuse. Tiles are only used when the frame is larger than `min_side`."""

    def __init__(self, inner: Detector, tile: int = 640, overlap: float = 0.2, min_side: int = 1000,
                 full_frame: bool = True, iou_thr: float = 0.5):
        self.inner = inner
        self.tile, self.overlap, self.min_side, self.full_frame, self.iou_thr = tile, overlap, min_side, full_frame, iou_thr
        self.labels = set(inner.labels)
        self.name = f"tiled({inner.name},{tile})"

    def _tiles(self, W: int, H: int):
        step = int(self.tile * (1 - self.overlap))
        xs = list(range(0, max(1, W - self.tile) + 1, step))
        ys = list(range(0, max(1, H - self.tile) + 1, step))
        if xs[-1] + self.tile < W:
            xs.append(W - self.tile)
        if ys[-1] + self.tile < H:
            ys.append(H - self.tile)
        return [(x, y) for y in ys for x in xs]

    def detect(self, frames: Sequence[np.ndarray]) -> list[list[Detection]]:
        out = []
        for f in frames:
            H, W = f.shape[:2]
            dets: list[Detection] = []
            if self.full_frame:
                dets += self.inner.detect([f])[0]
            if max(W, H) >= self.min_side and min(W, H) > self.tile // 2:
                tiles = self._tiles(W, H)
                crops = [f[y:y + self.tile, x:x + self.tile] for x, y in tiles]
                for (x, y), crop in zip(tiles, self.inner.detect(crops)):
                    th, tw = crops[0].shape[:2] if crops else (self.tile, self.tile)
                    for d in crop:
                        # tile box -> frame box
                        ch, cw = f[y:y + self.tile, x:x + self.tile].shape[:2]
                        bx = (x + d.bbox.x * cw) / W
                        by = (y + d.bbox.y * ch) / H
                        bw = d.bbox.w * cw / W
                        bh = d.bbox.h * ch / H
                        dets.append(Detection(BBox(bx, by, bw, bh).clamp(), d.label, d.score, d.source or self.inner.name))
            out.append(nms_merge(dets, self.iou_thr))
        return out


class MultiScaleDetector(Detector):
    """Run the inner detector at several input scales (upscaling helps tiny faces,
    downscaling helps very large ones) and fuse."""

    def __init__(self, inner: Detector, scales: Sequence[float] = (1.0, 1.5), iou_thr: float = 0.5, extra_every: int = 1):
        # extra_every: run the non-1.0 scales only on every k-th call (distant faces move slowly;
        # the tracker/carry bridges the steps in between) - halves the multi-scale cost at k=2.
        self.inner, self.scales, self.iou_thr = inner, tuple(scales), iou_thr
        self.extra_every = max(1, extra_every)
        self._calls = 0
        self.labels = set(inner.labels)
        self.name = f"multiscale({inner.name},{','.join(str(s) for s in scales)})"

    def detect(self, frames: Sequence[np.ndarray]) -> list[list[Detection]]:
        out = []
        for f in frames:
            self._calls += 1
            run_extra = (self._calls % self.extra_every) == 0
            dets: list[Detection] = []
            for s in self.scales:
                if abs(s - 1.0) > 1e-6 and not run_extra:
                    continue
                img = f if abs(s - 1.0) < 1e-6 else cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR if s > 1 else cv2.INTER_AREA)
                dets += self.inner.detect([img])[0]      # boxes are normalized: scale-invariant
            out.append(nms_merge(dets, self.iou_thr))
        return out


class EnsembleDetector(Detector):
    """Union of several detectors with class-aware NMS/box fusion."""

    def __init__(self, members: Sequence[Detector], iou_thr: float = 0.5):
        self.members = list(members)
        self.iou_thr = iou_thr
        self.labels = set().union(*(m.labels for m in self.members)) if self.members else set()
        self.name = "ensemble(" + ",".join(m.name for m in self.members) + ")"

    def detect(self, frames: Sequence[np.ndarray]) -> list[list[Detection]]:
        per_member = [m.detect(frames) for m in self.members]
        out = []
        for i in range(len(frames)):
            dets = []
            for m, res in zip(self.members, per_member):
                for d in res[i]:
                    dets.append(Detection(d.bbox, d.label, d.score, d.source or m.name))
            out.append(nms_merge(dets, self.iou_thr))
        return out

    def close(self) -> None:
        for m in self.members:
            m.close()


class HeadFallback(Detector):
    """When a person is detected but no face box lies in the head region, emit a
    `head` box covering the top of the person (hair, ears, profile, masked faces,
    motion blur, low light). Also enlarges every face box by `face_pad`.

    Head region heuristic (upright person): width = 45 % of person width centred,
    height = 22 % of person height from the top. For wide (lying / crouched) person
    boxes the heuristic is skipped; a dedicated head detector is the Phase C plan."""

    def __init__(self, inner: Detector, face_pad: float = 0.0, head_w: float = 0.45, head_h: float = 0.22,
                 min_person_h: float = 0.12, as_face: bool = True, emit_person: bool = True):
        # as_face: label head boxes "face" (source "head-fallback") so a person whose face is
        # seen in some frames and not others stays ONE track instead of alternating face/head tracks.
        # emit_person: pass the whole-body person boxes through as detections. The person class is
        # added to the object detector only to derive head boxes, so when the user did not ask for
        # "person" these boxes are review burden (422 whole-body tracks on a 45-min reference clip).
        self.inner = inner
        self.face_pad, self.head_w, self.head_h, self.min_person_h = face_pad, head_w, head_h, min_person_h
        self.as_face = as_face
        self.emit_person = emit_person
        self.labels = set(inner.labels) | ({"face"} if as_face else {"head"})
        self.name = f"headfallback({inner.name})"

    def detect(self, frames: Sequence[np.ndarray]) -> list[list[Detection]]:
        out = []
        for dets in self.inner.detect(frames):
            faces = [d for d in dets if d.label == "face"]
            res: list[Detection] = []
            for d in dets:
                if d.label == "face" and self.face_pad > 0:
                    b = d.bbox
                    px, py = b.w * self.face_pad, b.h * self.face_pad
                    res.append(Detection(BBox(b.x - px, b.y - py, b.w + 2 * px, b.h + 2 * py).clamp(), d.label, d.score, d.source))
                elif d.label != "person":
                    res.append(d)
                else:
                    if self.emit_person:
                        res.append(d)
                    p = d.bbox
                    if p.h < self.min_person_h or p.w > 1.3 * p.h:
                        continue
                    head = BBox(p.x + p.w * (1 - self.head_w) / 2, p.y, p.w * self.head_w, p.h * self.head_h).clamp()
                    if not any(_contains(head, f.bbox, 0.5) or _contains(f.bbox, head, 0.5) or head.iou(f.bbox) > 0.2 for f in faces):
                        res.append(Detection(head, "face" if self.as_face else "head", d.score * 0.9, "head-fallback"))
            out.append(res)
        return out
