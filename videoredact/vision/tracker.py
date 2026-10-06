"""Single-object tracking with loss detection and re-acquisition.

Workflow: the user draws a box on one frame. We
  1. build an appearance model (HSV histogram + grayscale template),
  2. track forward with OpenCV CSRT on a downscaled frame,
  3. detect loss (CSRT failure, appearance drift, or box leaving the frame),
  4. while lost, search every frame for the object using (a) a class
     detector if the box was labelled face/screen/etc. and (b) multi-scale
     template matching; a candidate is accepted when its appearance matches,
  5. re-initialise CSRT on the candidate and open a new visibility span,
  6. optionally run the same procedure backward from the start frame so an
     object that was already visible before the user drew the box is covered.

Everything runs on CPU. Work frames are capped at TRACK_MAX_SIDE pixels.
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

TRACK_MAX_SIDE = 640
HIST_BINS = (16, 16)       # H, S
LOST_AFTER_FRAMES = 4      # consecutive bad frames before declaring loss
HIST_DRIFT_THR = 0.55      # Bhattacharyya distance -> lost
HIST_ACCEPT_THR = 0.35     # Bhattacharyya distance -> re-acquire
TEMPLATE_ACCEPT = 0.62     # normalized cross-correlation -> re-acquire
SEARCH_SCALES = (0.7, 0.85, 1.0, 1.2, 1.45)
TEMPLATE_UPDATE_RATE = 0.0  # keep original template (safer against drift)


def _make_tracker():
    for ctor in ("TrackerCSRT_create",):
        if hasattr(cv2, ctor):
            return getattr(cv2, ctor)()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    raise RuntimeError("No OpenCV tracker available (install opencv-contrib-python)")


@dataclass
class Appearance:
    hist: np.ndarray
    template: np.ndarray            # grayscale patch at work scale
    size: tuple[int, int]           # (w, h) in work pixels at init
    label: Optional[str] = None     # detector class if known

    @staticmethod
    def build(work: np.ndarray, box: tuple[int, int, int, int], label: Optional[str]) -> "Appearance":
        x, y, w, h = box
        patch = work[y:y + h, x:x + w]
        return Appearance(_hist(patch), cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), (w, h), label)

    def distance(self, work: np.ndarray, box: tuple[int, int, int, int]) -> float:
        x, y, w, h = box
        if w < 2 or h < 2:
            return 1.0
        patch = work[y:y + h, x:x + w]
        if patch.size == 0:
            return 1.0
        return float(cv2.compareHist(self.hist, _hist(patch), cv2.HISTCMP_BHATTACHARYYA))


def _hist(patch: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, list(HIST_BINS), [0, 180, 0, 256])
    cv2.normalize(h, h, 0, 1, cv2.NORM_MINMAX)
    return h


def _clip_box(box, W, H) -> Optional[tuple[int, int, int, int]]:
    x, y, w, h = box
    x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
    x1, y1 = min(W, int(round(x + w))), min(H, int(round(y + h)))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    return x0, y0, x1 - x0, y1 - y0


def _search_template(work_gray: np.ndarray, app: Appearance) -> Optional[tuple[tuple[int, int, int, int], float]]:
    """Multi-scale template matching. Returns (box, score) of the best hit."""
    best = None
    H, W = work_gray.shape
    tw, th = app.size
    for s in SEARCH_SCALES:
        w, h = int(tw * s), int(th * s)
        if w < 8 or h < 8 or w >= W or h >= H:
            continue
        tpl = cv2.resize(app.template, (w, h), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(work_gray, tpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        if best is None or mx > best[1]:
            best = ((loc[0], loc[1], w, h), float(mx))
    return best


class ObjectTracker:
    """Tracks one object through a FrameReader; produces a VideoTrack."""

    def __init__(self, reader: FrameReader, detector: Optional[CombinedDetector] = None):
        self.reader = reader
        self.detector = detector
        W, H = reader.width, reader.height
        self.scale = min(1.0, TRACK_MAX_SIDE / max(W, H))
        self.wW, self.wH = int(round(W * self.scale)), int(round(H * self.scale))

    # ---- helpers ---------------------------------------------------------
    def _work(self, frame: np.ndarray) -> np.ndarray:
        if self.scale < 1.0:
            return cv2.resize(frame, (self.wW, self.wH), interpolation=cv2.INTER_AREA)
        return frame

    def _to_norm(self, box: tuple[int, int, int, int]) -> BBox:
        x, y, w, h = box
        return BBox(x / self.wW, y / self.wH, w / self.wW, h / self.wH).clamp()

    def _from_norm(self, bb: BBox) -> tuple[int, int, int, int]:
        return _clip_box((bb.x * self.wW, bb.y * self.wH, bb.w * self.wW, bb.h * self.wH), self.wW, self.wH)

    def _search(self, work: np.ndarray, app: Appearance) -> Optional[tuple[int, int, int, int]]:
        cands: list[tuple[tuple[int, int, int, int], float]] = []
        if self.detector and app.label:
            for d in self.detector.detect(work):
                if d.label != app.label:
                    continue
                b = self._from_norm(d.bbox)
                if b is None:
                    continue
                # size sanity: within 0.4x..2.5x of original
                ratio = (b[2] * b[3]) / max(1, app.size[0] * app.size[1])
                if 0.16 <= ratio <= 6.25:
                    cands.append((b, app.distance(work, b)))
            if cands:
                cands.sort(key=lambda c: c[1])
                if cands[0][1] < HIST_ACCEPT_THR + 0.1:   # detector hits get a bit more slack
                    return cands[0][0]
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        hit = _search_template(gray, app)
        if hit and hit[1] >= TEMPLATE_ACCEPT:
            b = _clip_box(hit[0], self.wW, self.wH)
            if b and app.distance(work, b) < HIST_ACCEPT_THR:
                return b
        return None

    # ---- main loop -------------------------------------------------------
    def _run_direction(self, frames, app: Appearance, init_box, spans: list[Span],
                       progress: Optional[ProgressCB], cancel: Optional[CancelCB],
                       total: int, label: str) -> None:
        """frames: iterable of (idx, frame) in processing order (forward or backward)."""
        tracker = _make_tracker()
        state = "init"
        bad = 0
        cur_span: Optional[Span] = None
        done = 0
        recent: list[int] = []
        for idx, frame in frames:
            if cancel and cancel():
                break
            work = self._work(frame)
            done += 1
            if progress and done % 10 == 0:
                progress(min(0.999, done / max(1, total)), f"{label} frame {idx}")
            if state == "init":
                tracker.init(work, tuple(int(v) for v in init_box))
                cur_span = Span(idx, idx, {idx: self._to_norm(init_box)})
                spans.append(cur_span)
                recent = [idx]
                state = "tracking"
                continue
            if state == "tracking":
                ok, box = tracker.update(work)
                b = _clip_box(box, self.wW, self.wH) if ok else None
                drift = app.distance(work, b) if b else 1.0
                if b is None or drift > HIST_DRIFT_THR:
                    bad += 1
                else:
                    bad = 0
                if bad >= LOST_AFTER_FRAMES or (b is None and bad >= 2):
                    # Drop the uncertain frames we just added, then go searching.
                    # (keyframes were added in processing order, so trim the tail)
                    for k in recent[-bad:]:
                        cur_span.keyframes.pop(k, None)
                    keys = sorted(cur_span.keyframes)
                    if keys:
                        cur_span.start_frame, cur_span.end_frame = keys[0], keys[-1]
                    state = "lost"
                    bad = 0
                    recent = []
                    continue
                if b is not None:
                    cur_span.keyframes[idx] = self._to_norm(b)
                    cur_span.start_frame = min(cur_span.start_frame, idx)
                    cur_span.end_frame = max(cur_span.end_frame, idx)
                    recent.append(idx)
                    if len(recent) > 32:
                        recent = recent[-32:]
                continue
            if state == "lost":
                b = self._search(work, app)
                if b is not None:
                    tracker = _make_tracker()
                    tracker.init(work, b)
                    cur_span = Span(idx, idx, {idx: self._to_norm(b)})
                    spans.append(cur_span)
                    recent = [idx]
                    state = "tracking"
                    bad = 0

    def track(self, start_frame: int, bbox: BBox, label: Optional[str] = None,
              end_frame: Optional[int] = None, backward: bool = True,
              progress: Optional[ProgressCB] = None, cancel: Optional[CancelCB] = None) -> VideoTrack:
        reader = self.reader
        last = (end_frame if end_frame is not None else reader.frame_count - 1)
        first = reader.read_at(start_frame)
        if first is None:
            raise RuntimeError(f"Cannot read frame {start_frame}")
        work0 = self._work(first)
        init_box = self._from_norm(bbox)
        if init_box is None:
            raise ValueError("Box too small")
        app = Appearance.build(work0, init_box, label)
        spans: list[Span] = []

        total_fwd = max(0, last - start_frame + 1)
        total_bwd = start_frame if backward else 0
        total = total_fwd + total_bwd

        # forward
        def fwd():
            yield start_frame, first
            for idx, f in reader.iter_frames(start_frame + 1, last + 1):
                yield idx, f
        fwd_spans: list[Span] = []
        self._run_direction(fwd(), app, init_box, fwd_spans, progress, cancel, total, "Tracking forward")
        spans += fwd_spans

        # backward (chunked reverse reads)
        if backward and start_frame > 0 and not (cancel and cancel()):
            def bwd():
                yield start_frame, first
                CH = 48
                hi = start_frame  # exclusive
                while hi > 0:
                    lo = max(0, hi - CH)
                    chunk = list(reader.iter_frames(lo, hi))
                    for idx, f in reversed(chunk):
                        yield idx, f
                    hi = lo
            bwd_spans: list[Span] = []
            self._run_direction(bwd(), app, init_box, bwd_spans, progress, cancel, total, "Tracking backward")
            # spans built backward have start=end=start_frame grown downward; normalize
            for s in bwd_spans:
                keys = sorted(s.keyframes)
                if keys:
                    s.start_frame, s.end_frame = keys[0], keys[-1]
            spans += bwd_spans

        track = VideoTrack(label=label or "object", spans=[], source="tracked")
        for s in spans:
            keys = sorted(s.keyframes)
            if not keys:
                continue
            s.start_frame, s.end_frame = min(s.start_frame, keys[0]), max(s.end_frame, keys[-1])
            track.spans.append(s)
        track.merge_spans()
        # lead-in / lead-out: 2 frames each side guards against boundary error
        for s in track.spans:
            s.start_frame = max(0, s.start_frame - 2)
            s.end_frame = min(last, s.end_frame + 2)
        track.merge_spans()
        if progress:
            progress(1.0, "Tracking complete")
        return track
