"""Automatic detection pass: find faces / screens / documents / phones / people
and link them over time into VideoTracks.

Pipeline per chunk of frames (runs in parallel worker processes):
  * detectors run every `stride` frames on a <=960 px copy of the frame
    (faces at <=1280 px inside the face detector),
  * detections are matched to live tracks of the same label by IoU
    (greedy, ByteTrack-style two-pass: confident detections first),
  * a live track that misses a detection is *carried* with ViTTrack from its
    last detected box through every following frame (using a small ring
    buffer of recent frames to start from the exact detection frame) until a
    detection matches it again or the carry fails / drifts / times out.
    This keeps a face covered while it is in profile or motion-blurred, which
    IoU-only linking left exposed;
  * tracks that outlive `max_missed` detection steps are closed; very short
    tracks (< min_hits) are dropped as noise; every track gets lead-in/out.

Chunks overlap by a few detection steps and tracks are merged across chunk
boundaries by IoU. On an 8-core machine 3-4 workers give ~3x the throughput
of the single-process version.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np
import logging

from videoredact.core.media import FrameReader
from videoredact.core.model import BBox, Span, VideoTrack
from .detector import CombinedDetector, Detection

ProgressCB = Callable[[float, str], None]
CancelCB = Callable[[], bool]
log = logging.getLogger(__name__)

DETECT_MAX_SIDE = 960
CARRY_WORK_SIDE = 512
CARRY_MIN_SCORE = 0.35
CARRY_HIST_THR = 0.65
CARRY_MAX_S = 3.0          # give up carrying after this long without a detection
CARRY_STEP = 2             # update carriers every N-th frame (boxes interpolate in between)
CARRY_TRACKER = "vit"      # vit: 5 ms per face, scale-adaptive (90% coverage); kcf: 2 ms but drifts (84%)
CARRY_MIN_HITS = 2         # only carry tracks confirmed by >= 2 detections (avoids carrying false positives)
SINGLE_MIN_CONF = 0.0      # keep every single detection (recall first; the reviewer can delete)
LINK_IOU = 0.2             # fragment linking: overlap between the end box of one and start box of the next


@dataclass
class AutoDetectOptions:
    labels: list[str] = field(default_factory=lambda: ["face", "screen", "document"])
    stride: int = 5
    face_conf: float = 0.5
    obj_conf: float = 0.35
    obj_model: str = "yolox_tiny"
    iou_thr: float = 0.3
    max_missed: int = 8        # detection steps (stride frames each)
    min_hits: int = 1          # recall first: keep single detections too (if confident, see SINGLE_MIN_CONF)
    link_gap_s: float = 1.0    # post-pass: join same-label fragments separated by <= this gap when boxes overlap
    start_frame: int = 0
    end_frame: Optional[int] = None
    threads: int = 0
    face_model: str = "yunet"
    carry: bool = True         # carry tracks with ViTTrack through missed detections
    workers: int = 0           # 0 = auto (parallel chunks); 1 = single process


# ----------------------------------------------------------------------------
# single-chunk engine
# ----------------------------------------------------------------------------
class _Live:
    __slots__ = ("label", "span", "last_box", "last_frame", "hits", "missed", "score_sum", "carrier",
                 "carry_frames")

    def __init__(self, label: str, frame: int, box: BBox, score: float):
        self.label = label
        self.span = Span(frame, frame, {frame: box})
        self.last_box = box
        self.last_frame = frame
        self.hits = 1
        self.missed = 0
        self.score_sum = score
        self.carrier = None
        self.carry_frames = 0


def _match(live: list["_Live"], dets: list[Detection], iou_thr: float):
    if not live or not dets:
        return [], list(range(len(live))), list(range(len(dets)))
    iou = np.zeros((len(live), len(dets)), dtype=np.float32)
    for i, L in enumerate(live):
        for j, d in enumerate(dets):
            if L.label == d.label:
                iou[i, j] = L.last_box.iou(d.bbox)
    pairs, used_i, used_j = [], set(), set()
    # two passes: confident detections first (ByteTrack idea), then the rest
    for conf_min in (0.5, 0.0):
        flat = sorted(((iou[i, j], i, j) for i in range(len(live)) for j in range(len(dets))
                       if dets[j].score >= conf_min), reverse=True)
        for v, i, j in flat:
            if v < iou_thr:
                break
            if i in used_i or j in used_j:
                continue
            pairs.append((i, j))
            used_i.add(i)
            used_j.add(j)
    return pairs, [i for i in range(len(live)) if i not in used_i], [j for j in range(len(dets)) if j not in used_j]


def _hist(patch: np.ndarray):
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(h, h, 0, 1, cv2.NORM_MINMAX)
    return h


class _Carrier:
    """Runs a ViTTrack (or fallback) tracker for one track between detections."""

    def __init__(self, work_frames: list[np.ndarray], box: BBox, W: int, H: int, kind: str = CARRY_TRACKER):
        from .tracker import make_tracker, tracker_score, vit_available
        self.W, self.H = W, H
        self.tracker = make_tracker(kind)
        self.is_vit = kind == "vit" and vit_available()
        self._score = tracker_score
        x0, y0, x1, y1 = box.to_pixels(W, H)
        self.tracker.init(work_frames[0], (x0, y0, max(4, x1 - x0), max(4, y1 - y0)))
        self.hist = _hist(work_frames[0][y0:y1, x0:x1]) if (y1 - y0) > 2 and (x1 - x0) > 2 else None
        self.box: Optional[BBox] = box
        for f in work_frames[1:]:
            self.update(f)

    def update(self, work: np.ndarray) -> Optional[BBox]:
        ok, b = self.tracker.update(work)
        if not ok or (self.is_vit and self._score(self.tracker) < CARRY_MIN_SCORE):
            return None
        x, y, w, h = (int(round(v)) for v in b)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.W, x + w), min(self.H, y + h)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        if self.hist is not None:
            d = cv2.compareHist(self.hist, _hist(work[y0:y1, x0:x1]), cv2.HISTCMP_BHATTACHARYYA)
            if d > CARRY_HIST_THR:
                return None
        self.box = BBox.from_pixels(x0, y0, x1, y1, self.W, self.H)
        return self.box


def _detect_chunk(reader: FrameReader, opts: AutoDetectOptions, det: CombinedDetector,
                  start: int, end: int, progress: Optional[ProgressCB], cancel: Optional[CancelCB],
                  prog_base: float = 0.0, prog_span: float = 1.0) -> list[VideoTrack]:
    W, H = reader.width, reader.height
    stride = max(1, opts.stride)
    dscale = min(1.0, DETECT_MAX_SIDE / max(W, H))
    dsize = (int(round(W * dscale)), int(round(H * dscale)))
    cscale = min(1.0, CARRY_WORK_SIDE / max(W, H))
    csize = (int(round(W * cscale)), int(round(H * cscale)))
    ring: deque = deque(maxlen=stride + 1)   # (idx, original frame) - resized lazily when a carrier starts

    def to_work(f):
        return cv2.resize(f, csize, interpolation=cv2.INTER_LINEAR) if cscale < 1 else f
    live: list[_Live] = []
    finished: list[_Live] = []
    total = max(1, end - start)
    fps = reader.fps or 30.0
    carry_max_frames = int(CARRY_MAX_S * fps)

    def close(L: _Live):
        if L.hits >= max(2, opts.min_hits) or (opts.min_hits <= 1 and L.hits == 1 and L.score_sum >= SINGLE_MIN_CONF):
            finished.append(L)

    reader.seek(start)
    idx = start
    t_last = time.time()
    while idx < end:
        if cancel and cancel():
            break
        frame = reader.read()
        if frame is None:
            break
        is_det = (idx - start) % stride == 0
        carrying = opts.carry and any(L.carrier is not None for L in live)
        work_c = None
        if opts.carry:
            ring.append((idx, frame))
        if is_det:
            # full-resolution frame: the face detector downsizes to <=1280 itself (small faces
            # need the pixels; a 960 px copy lost ~7 % of faces), YOLOX resizes to 416 anyway
            dets = det.detect(frame)
            pairs, un_live, un_det = _match(live, dets, opts.iou_thr)
            for i, j in pairs:
                L, d = live[i], dets[j]
                L.span.keyframes[idx] = d.bbox
                L.span.end_frame = idx
                L.last_box, L.last_frame = d.bbox, idx
                L.hits += 1
                L.missed = 0
                L.score_sum += d.score
                L.carrier = None
                L.carry_frames = 0
            for i in un_live:
                L = live[i]
                L.missed += 1
                if opts.carry and L.carrier is None and L.missed == 1 and L.hits >= CARRY_MIN_HITS and ring:
                    frames = [to_work(f) for k, f in ring if k >= L.last_frame and (k - L.last_frame) % CARRY_STEP == 0]
                    if frames:
                        try:
                            L.carrier = _Carrier(frames, L.last_box, csize[0], csize[1])
                            if L.carrier.box is not None:
                                L.last_box = L.carrier.box
                                L.span.keyframes[idx] = L.carrier.box
                                L.span.end_frame = idx
                        except Exception:
                            L.carrier = None
            for j in un_det:
                d = dets[j]
                live.append(_Live(d.label, idx, d.bbox, d.score))
            still = []
            for L in live:
                if L.missed > opts.max_missed:
                    close(L)
                else:
                    still.append(L)
            live = still
            if progress:
                now = time.time()
                if now - t_last > 0.4:
                    t_last = now
                    progress(prog_base + prog_span * min(0.999, (idx - start) / total),
                             f"Detecting frame {idx}/{end}  ({len(live)} live, {len(finished)} done)")
        elif carrying and (idx - start) % CARRY_STEP == 0:
            work_c = to_work(frame)
            for L in live:
                if L.carrier is not None:
                    b = L.carrier.update(work_c)
                    L.carry_frames += 1
                    if b is None or L.carry_frames > carry_max_frames:
                        L.carrier = None
                        continue
                    L.span.keyframes[idx] = b
                    L.span.end_frame = idx
                    L.last_box = b
        idx += 1
    for L in live:
        close(L)
    tracks: list[VideoTrack] = []
    last_frame = end - 1
    for L in finished:
        s = L.span
        s.start_frame = max(0, s.start_frame - stride)
        s.end_frame = min(last_frame, s.end_frame + stride)
        note = f"auto-detected, {L.hits} hits, avg conf {L.score_sum / L.hits:.2f}"
        if L.hits == 1:
            note += " (single detection - review)"
        tracks.append(VideoTrack(label=L.label, spans=[s], source="auto", note=note))
    return link_fragments(tracks, int(opts.link_gap_s * fps))


def _near(a: BBox, b: BBox, gap: int, max_gap: int) -> bool:
    """Boxes of similar size whose centres are within ~1.5 box-sizes (scaled by how
    short the gap is): plausible continuation of the same object."""
    sa, sb = max(a.w, a.h), max(b.w, b.h)
    if not (0.5 <= sa / max(sb, 1e-6) <= 2.0):
        return False
    (ax, ay), (bx, by) = a.center(), b.center()
    dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    allow = max(sa, sb) * (0.8 + 0.7 * gap / max(1, max_gap))
    return dist <= allow


def link_fragments(tracks: list[VideoTrack], max_gap: int, iou_thr: float = LINK_IOU) -> list[VideoTrack]:
    """Join same-label tracks whose gap is <= max_gap frames and whose boxes at the
    boundary overlap. The gap is then covered by interpolation. Cuts the number of
    fragments a reviewer has to look at and covers short detection dropouts."""
    tracks = sorted(tracks, key=lambda t: t.first_frame())
    out: list[VideoTrack] = []
    for t in tracks:
        tb = t.bbox_at(t.first_frame())
        best = None
        for m in out:
            if m.label != t.label or tb is None:
                continue
            gap = t.first_frame() - m.last_frame()
            if 0 < gap <= max_gap:
                mb = m.bbox_at(m.last_frame())
                if mb is not None and (mb.iou(tb) >= iou_thr or _near(mb, tb, gap, max_gap)):
                    if best is None or m.last_frame() > best.last_frame():
                        best = m
        if best is not None:
            best.spans.extend(t.spans)
            best.merge_spans()
            # bridge the gap so interpolation covers it
            lo, hi = sorted(best.spans, key=lambda s: s.start_frame)[0].start_frame, best.last_frame()
            best.spans = [Span(lo, hi, {k: v for s in best.spans for k, v in s.keyframes.items()})]
            best.note += "; linked"
        else:
            out.append(t)
    return out


# ----------------------------------------------------------------------------
# parallel driver
# ----------------------------------------------------------------------------
def _worker(path: str, opts_d: dict, start: int, end: int, q, cancel_ev, wid: int) -> None:
    try:
        cv_threads = opts_d.pop("cv_threads", 2)
        opts = AutoDetectOptions(**opts_d)
        cv2.setNumThreads(max(2, cv_threads))
        det = CombinedDetector(opts.labels, opts.face_conf, opts.obj_conf, opts.obj_model,
                               threads=max(1, opts.threads), face_model=opts.face_model)
        t0 = time.time()
        with FrameReader(path) as reader:
            tracks = _detect_chunk(reader, opts, det, start, end,
                                   progress=lambda p, m: q.put(("progress", wid, p, m)),
                                   cancel=lambda: cancel_ev.is_set())
        q.put(("done", wid, [t.to_dict() for t in tracks], f"{end - start} frames in {time.time() - t0:.1f}s"))
    except Exception as e:  # noqa: BLE001
        import traceback
        q.put(("error", wid, None, f"{e}\n{traceback.format_exc()}"))


def _merge_chunks(chunks: list[list[VideoTrack]], iou_thr: float = 0.3) -> list[VideoTrack]:
    """Join tracks that continue across a chunk boundary (overlapping frames, same label, IoU)."""
    merged: list[VideoTrack] = list(chunks[0]) if chunks else []
    for nxt in chunks[1:]:
        used: set[int] = set()
        for t in nxt:
            best, best_iou = None, iou_thr
            for m in merged:
                if m.label != t.label or id(m) in used:
                    continue
                lo, hi = max(m.first_frame(), t.first_frame()), min(m.last_frame(), t.last_frame())
                if hi < lo:
                    continue
                for f in range(lo, hi + 1, max(1, (hi - lo) // 4 or 1)):
                    a, b = m.bbox_at(f), t.bbox_at(f)
                    if a is not None and b is not None:
                        v = a.iou(b)
                        if v > best_iou:
                            best, best_iou = m, v
                        break
            if best is not None:
                used.add(id(best))
                best.spans.extend(t.spans)
                best.merge_spans()
                best.note += "; merged across chunks"
            else:
                merged.append(t)
    return merged


def run_auto_detect(reader: FrameReader, opts: AutoDetectOptions,
                    progress: Optional[ProgressCB] = None,
                    cancel: Optional[CancelCB] = None,
                    detector: Optional[CombinedDetector] = None) -> list[VideoTrack]:
    """Detect + link over [start_frame, end_frame). Uses parallel worker processes
    when the range is long enough and opts.workers != 1."""
    start = opts.start_frame
    end = opts.end_frame if opts.end_frame is not None else reader.frame_count
    n = end - start
    workers = opts.workers or max(1, min(4, (os.cpu_count() or 4) // 2))
    if detector is not None or workers <= 1 or n < 1500:
        det = detector or CombinedDetector(opts.labels, opts.face_conf, opts.obj_conf, opts.obj_model,
                                           opts.threads, face_model=opts.face_model)
        tracks = _detect_chunk(reader, opts, det, start, end, progress, cancel)
        tracks.sort(key=lambda t: (t.first_frame(), t.label))
        if progress:
            progress(1.0, f"Detection complete: {len(tracks)} regions")
        return tracks

    # ---- parallel ------------------------------------------------------------
    overlap = opts.stride * 3
    size = n // workers
    bounds = []
    for w in range(workers):
        a = start + w * size
        b = end if w == workers - 1 else min(end, start + (w + 1) * size + overlap)
        bounds.append((a, b))
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    cancel_ev = ctx.Event()
    opts_d = asdict(opts)
    cpu = os.cpu_count() or 4
    opts_d["threads"] = max(1, (opts.threads or cpu) // workers)
    opts_d["workers"] = 1
    cv_threads = max(2, cpu // workers)
    procs = []
    for wid, (a, b) in enumerate(bounds):
        p = ctx.Process(target=_worker, args=(reader.path, dict(opts_d, cv_threads=cv_threads), a, b, q, cancel_ev, wid), daemon=True)
        p.start()
        procs.append(p)
    results: dict[int, list[VideoTrack]] = {}
    prog = [0.0] * workers
    if progress:
        progress(0.0, f"Detecting with {workers} parallel workers…")
    try:
        while len(results) < workers:
            if cancel and cancel():
                cancel_ev.set()
                break
            try:
                kind, wid, a, b = q.get(timeout=0.25)
            except Exception:
                if all(not p.is_alive() for p in procs) and len(results) < workers:
                    raise RuntimeError("A detection worker exited unexpectedly (see log)")
                continue
            if kind == "progress":
                prog[wid] = a
                if progress:
                    progress(sum(prog) / workers, f"Detecting ({workers} workers): " + b.split("  (", 1)[0])
            elif kind == "done":
                results[wid] = [VideoTrack.from_dict(d) for d in a]
                log.info("detect worker %d: %s", wid, b)
            elif kind == "error":
                raise RuntimeError(f"Detection worker failed: {b}")
    finally:
        cancel_ev.set()
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
    if cancel and cancel():
        return []
    tracks = _merge_chunks([results[w] for w in range(workers)], opts.iou_thr)
    tracks.sort(key=lambda t: (t.first_frame(), t.label))
    if progress:
        progress(1.0, f"Detection complete: {len(tracks)} regions")
    return tracks
