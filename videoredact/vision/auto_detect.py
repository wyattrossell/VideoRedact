"""Automatic detection pass: find faces / screens / documents / people in a
video and link detections over time into VideoTracks.

Association is a lightweight ByteTrack-style greedy IoU matcher (our own
implementation; the MIT-licensed ByteTrack paper/algorithm, no AGPL code):
  * detectors run every `stride` frames on a downscaled frame,
  * each detection is matched to the live track of the same label with the
    highest IoU (>= iou_thr); unmatched detections start new tracks,
  * a track survives `max_missed` detection steps without a match (its box
    is interpolated across the gap), then is closed,
  * tracks shorter than `min_hits` detections are dropped as noise,
  * every track gets lead-in/lead-out padding of one stride.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from videoredact.core.media import FrameReader
from videoredact.core.model import BBox, Span, VideoTrack
from .detector import CombinedDetector, Detection

ProgressCB = Callable[[float, str], None]
CancelCB = Callable[[], bool]

DETECT_MAX_SIDE = 960


@dataclass
class _Live:
    label: str
    span: Span
    last_box: BBox
    last_frame: int
    hits: int = 1
    missed: int = 0
    score_sum: float = 0.0


@dataclass
class AutoDetectOptions:
    labels: list[str] = field(default_factory=lambda: ["face", "screen", "document"])
    stride: int = 3
    face_conf: float = 0.6
    obj_conf: float = 0.35
    obj_model: str = "yolox_s"
    iou_thr: float = 0.3
    max_missed: int = 8        # detection steps (stride frames each)
    min_hits: int = 2
    start_frame: int = 0
    end_frame: Optional[int] = None
    threads: int = 0
    face_model: str = "yunet"


def _match(live: list[_Live], dets: list[Detection], iou_thr: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy IoU matching per label. Returns (pairs, unmatched_live, unmatched_dets)."""
    if not live or not dets:
        return [], list(range(len(live))), list(range(len(dets)))
    iou = np.zeros((len(live), len(dets)), dtype=np.float32)
    for i, L in enumerate(live):
        for j, d in enumerate(dets):
            if L.label == d.label:
                iou[i, j] = L.last_box.iou(d.bbox)
    pairs = []
    used_i, used_j = set(), set()
    flat = sorted(((iou[i, j], i, j) for i in range(len(live)) for j in range(len(dets))), reverse=True)
    for v, i, j in flat:
        if v < iou_thr:
            break
        if i in used_i or j in used_j:
            continue
        pairs.append((i, j))
        used_i.add(i)
        used_j.add(j)
    return pairs, [i for i in range(len(live)) if i not in used_i], [j for j in range(len(dets)) if j not in used_j]


def run_auto_detect(reader: FrameReader, opts: AutoDetectOptions,
                    progress: Optional[ProgressCB] = None,
                    cancel: Optional[CancelCB] = None,
                    detector: Optional[CombinedDetector] = None) -> list[VideoTrack]:
    det = detector or CombinedDetector(opts.labels, opts.face_conf, opts.obj_conf, opts.obj_model, opts.threads,
                                       face_model=opts.face_model)
    W, H = reader.width, reader.height
    scale = min(1.0, DETECT_MAX_SIDE / max(W, H))
    size = (int(round(W * scale)), int(round(H * scale)))
    start = opts.start_frame
    end = opts.end_frame if opts.end_frame is not None else reader.frame_count
    stride = max(1, opts.stride)
    live: list[_Live] = []
    finished: list[_Live] = []
    total = max(1, end - start)

    def close(L: _Live) -> None:
        if L.hits >= opts.min_hits:
            finished.append(L)

    reader.seek(start)
    idx = start
    while idx < end:
        if cancel and cancel():
            break
        frame = reader.read()
        if frame is None:
            break
        if (idx - start) % stride == 0:
            work = cv2.resize(frame, size, interpolation=cv2.INTER_AREA) if scale < 1 else frame
            dets = det.detect(work)
            pairs, un_live, un_det = _match(live, dets, opts.iou_thr)
            for i, j in pairs:
                L, d = live[i], dets[j]
                L.span.keyframes[idx] = d.bbox
                L.span.end_frame = idx
                L.last_box, L.last_frame = d.bbox, idx
                L.hits += 1
                L.missed = 0
                L.score_sum += d.score
            for i in un_live:
                live[i].missed += 1
            for j in un_det:
                d = dets[j]
                live.append(_Live(d.label, Span(idx, idx, {idx: d.bbox}), d.bbox, idx, 1, 0, d.score))
            still = []
            for L in live:
                if L.missed > opts.max_missed:
                    close(L)
                else:
                    still.append(L)
            live = still
            if progress and ((idx - start) // stride) % 5 == 0:
                progress(min(0.999, (idx - start) / total),
                         f"Detecting frame {idx}/{end}  ({len(live)} live, {len(finished)} done)")
        idx += 1
    for L in live:
        close(L)

    tracks: list[VideoTrack] = []
    last_frame = end - 1
    for L in finished:
        s = L.span
        s.start_frame = max(0, s.start_frame - stride)
        s.end_frame = min(last_frame, s.end_frame + stride)
        t = VideoTrack(label=L.label, spans=[s], source="auto",
                       note=f"auto-detected, {L.hits} hits, avg conf {L.score_sum / L.hits:.2f}")
        tracks.append(t)
    tracks.sort(key=lambda t: (t.first_frame(), t.label))
    if progress:
        progress(1.0, f"Detection complete: {len(tracks)} regions")
    return tracks
