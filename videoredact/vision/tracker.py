"""Single-object tracking with loss detection and re-acquisition - fast version.

Measured on real 720p body-cam footage (i7-14700T, CPU only):
  CSRT 72 ms/frame, KCF 3.4 ms, MOSSE 0.2 ms, decode 1.3 ms, 5-scale template
  search @512 px 12 ms, YOLOX-tiny 26 ms, YuNet 51 ms.
CSRT alone made a 14-minute clip take ~30 minutes to track, which is unusable.

Design:
  * KCF on a <=512 px work frame, processing every `step`-th frame (default 2)
    and interpolating in between.
  * Drift / scale correction every `correct_every` processed frames: a local
    multi-scale template match of the original patch around the current box;
    and, when the object has a detector class (face/screen/person/phone/
    document), snap to the overlapping detection every `detect_every` frames.
  * Loss detection: tracker failure, appearance (HSV histogram) drift for a few
    consecutive frames, or the box leaving the frame.
  * Lost mode: whole-frame multi-scale template search on every other
    processed frame plus the class detector every 5th; accept when the
    appearance matches, re-initialise and open a new visibility span.
  * Backward pass from the drawn box (same engine, chunked reverse reads) so
    an object already on screen before the user drew the box is covered; it
    stops after the object has been gone for `backward_lost_limit_s`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import cv2
import numpy as np

from videoredact.core.media import FrameReader
from videoredact.core.model import BBox, Span, VideoTrack
from .detector import CombinedDetector

ProgressCB = Callable[[float, str], None]
CancelCB = Callable[[], bool]
UpdateCB = Callable[[list[Span]], None]   # receives a snapshot of the spans found so far
UPDATE_EVERY = 45                          # processed frames between live snapshots

HIST_BINS = (16, 16)
LOST_AFTER_BAD = 3
HIST_DRIFT_THR = 0.60
HIST_ACCEPT_THR = 0.38
TEMPLATE_ACCEPT = 0.60
LOCAL_CORRECT_MIN = 0.50
SEARCH_SCALES = (0.7, 0.85, 1.0, 1.2, 1.45)
LOCAL_SCALES = (0.82, 0.91, 1.0, 1.1, 1.22)
DETECTOR_LABELS = {"face", "screen", "document", "phone", "person"}
TARGET_OBJ_PX = 110          # desired object size on the work frame
MIN_WORK_SIDE = 256
REINIT_IOU = 0.85            # re-create the tracker only if a correction moved the box this much
VIT_MIN_SCORE = 0.30         # ViTTrack confidence below this counts as a failed update (classic engine)
VIT_RAW_THRESHOLD = 0.05     # OpenCV-internal failure cut for ViTTrack (default 0.20); see make_tracker


@dataclass
class TrackOptions:
    engine: str = "anchored"        # anchored (detector classes -> vision/anchored.py, the rest -> this loop) | anchored-all | classic
    step: int = 2                   # process every N-th frame
    work_side: int = 512            # longest side of the working frame (classic engine only)
    tracker: str = "vit"            # vit (default, scale-adaptive) | kcf (fastest) | csrt (slow)
    backward: bool = True
    lost_limit_s: float = 8.0       # stop (either direction) once the object has been gone this long
    search_to_end: bool = False     # forward: keep searching to end_frame even after lost_limit_s (object leaves and returns)
    end_frame: Optional[int] = None
    correct_every: int = 12         # processed frames between local template corrections
    detect_every: int = 10          # processed frames between detector snaps (tracking)
    lost_search_every: int = 3      # processed frames between template searches (lost)
    lost_detect_every: int = 8      # processed frames between detector searches (lost)
    pad_frames: int = 2
    # A manually drawn box is a statement that the object must be covered: hold the last box for a
    # moment after the tracker loses it and bridge short gaps between re-acquisitions, so the
    # redaction does not flicker (a dark notepad on the reference clip came back as 17 fragments
    # covering 13 of 22 s before this; see bench/results/PRECISION_PROFILE.md).
    hold_frames: int = 12           # frames the last box persists after a loss, both directions
    bridge_gap_s: float = 2.0       # gaps up to this long between nearby spans are interpolated over


_VIT_PATH: Optional[str] = None


def vit_available() -> bool:
    """ViTTrack (opencv_zoo, Apache-2.0): scale-adaptive deep tracker, ~5 ms/frame on CPU,
    measured 3x faster than KCF on large boxes and far more accurate on scale change."""
    global _VIT_PATH
    if _VIT_PATH is None:
        try:
            from . import models as reg
            _VIT_PATH = str(reg.ensure_model("vittrack")) if hasattr(cv2, "TrackerVit_create") else ""
        except Exception:
            _VIT_PATH = ""
    return bool(_VIT_PATH)


_POOL: dict[str, list] = {}


def release_tracker(tracker, kind: str = "vit") -> None:
    """Return a tracker to the pool so the next make_tracker() can re-init it
    (TrackerVit_create costs ~20 ms because it loads the ONNX; init() on an
    existing object is cheap). Pools are per process."""
    if tracker is not None and kind == "vit":
        pool = _POOL.setdefault(kind, [])
        if len(pool) < 32:
            pool.append(tracker)


def make_tracker(kind: str = "vit"):
    """Create (or reuse) an OpenCV tracker: vit (default) | kcf | csrt. Falls back to KCF."""
    if kind == "vit" and vit_available():
        pool = _POOL.get("vit")
        if pool:
            return pool.pop()
        p = cv2.TrackerVit_Params()
        p.net = _VIT_PATH
        # OpenCV's update() reports failure below 0.20 and then returns no box at all; the engines
        # apply their own score hysteresis (VIT_MIN_SCORE here, S_WEAK in anchored.py), so keep the
        # box coming and let them decide
        p.tracking_score_threshold = VIT_RAW_THRESHOLD
        return cv2.TrackerVit_create(p)
    if kind == "csrt" and hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if kind == "kcf" and hasattr(cv2, "TrackerKCF_create"):
        p = cv2.TrackerKCF_Params()
        p.detect_thresh = 0.4
        return cv2.TrackerKCF_create(p)
    # requested kind unavailable (KCF/CSRT live in opencv-contrib; a plain opencv-python
    # install shadows them) -> best available, ViT first
    if vit_available():
        p = cv2.TrackerVit_Params()
        p.net = _VIT_PATH
        return cv2.TrackerVit_create(p)
    if hasattr(cv2, "TrackerKCF_create"):
        p = cv2.TrackerKCF_Params()
        p.detect_thresh = 0.4
        return cv2.TrackerKCF_create(p)
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    raise RuntimeError("No OpenCV tracker available (install opencv-contrib-python)")


def tracker_score(tracker) -> float:
    """Confidence of the last update (ViT exposes one; others report 1.0)."""
    try:
        return float(tracker.getTrackingScore())
    except Exception:
        return 1.0


_make_tracker = make_tracker


def _hist(patch: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, list(HIST_BINS), [0, 180, 0, 256])
    cv2.normalize(h, h, 0, 1, cv2.NORM_MINMAX)
    return h


def _clip_box(box, W, H) -> Optional[tuple[int, int, int, int]]:
    x, y, w, h = box
    x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
    x1, y1 = min(W, int(round(x + w))), min(H, int(round(y + h)))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return x0, y0, x1 - x0, y1 - y0


@dataclass
class Appearance:
    hist: np.ndarray
    template: np.ndarray          # grayscale original patch (work scale)
    size: tuple[int, int]
    label: Optional[str]

    @staticmethod
    def build(work: np.ndarray, box, label) -> "Appearance":
        x, y, w, h = box
        patch = work[y:y + h, x:x + w]
        return Appearance(_hist(patch), cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), (w, h), label)

    def distance(self, work: np.ndarray, box) -> float:
        if box is None:
            return 1.0
        x, y, w, h = box
        patch = work[y:y + h, x:x + w]
        if patch.size == 0 or w < 2 or h < 2:
            return 1.0
        return float(cv2.compareHist(self.hist, _hist(patch), cv2.HISTCMP_BHATTACHARYYA))


def _match_scales(gray: np.ndarray, app: Appearance, scales, offset=(0, 0)):
    """Best multi-scale template match in `gray`. Returns (box, score) or None."""
    best = None
    H, W = gray.shape
    tw, th = app.size
    for s in scales:
        w, h = int(round(tw * s)), int(round(th * s))
        if w < 8 or h < 8 or w >= W or h >= H:
            continue
        tpl = cv2.resize(app.template, (w, h), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        if best is None or mx > best[1]:
            best = ((loc[0] + offset[0], loc[1] + offset[1], w, h), float(mx))
    return best


class ObjectTracker:
    def __init__(self, reader: FrameReader, detector: Optional[CombinedDetector] = None,
                 opts: Optional[TrackOptions] = None):
        self.reader = reader
        self.detector = detector
        self.o = opts or TrackOptions()
        self._set_work_side(self.o.work_side)
        self._on_update: Optional[UpdateCB] = None
        self._live: list[list[Span]] = []

    def _emit_update(self) -> None:
        if not self._on_update:
            return
        snap = [Span(s.start_frame, s.end_frame, dict(s.keyframes)) for lst in self._live for s in lst if s.keyframes]
        self._on_update(snap)

    def _set_work_side(self, side: int) -> None:
        W, H = self.reader.width, self.reader.height
        self.scale = min(1.0, side / max(W, H))
        self.wW, self.wH = int(round(W * self.scale)), int(round(H * self.scale))

    def _adapt_work_side(self, bbox: BBox) -> None:
        """KCF cost grows with the box area: shrink the work frame so the object is
        ~TARGET_OBJ_PX across (bounded), which keeps per-frame cost roughly constant."""
        W, H = self.reader.width, self.reader.height
        obj_px = max(bbox.w * W, bbox.h * H)
        if obj_px <= 0:
            return
        side = int(max(W, H) * TARGET_OBJ_PX / obj_px)
        self._set_work_side(max(MIN_WORK_SIDE, min(self.o.work_side, side)))

    # ---- helpers ---------------------------------------------------------
    def _work(self, frame: np.ndarray) -> np.ndarray:
        if self.scale < 1.0:
            return cv2.resize(frame, (self.wW, self.wH), interpolation=cv2.INTER_LINEAR)
        return frame

    def _to_norm(self, box) -> BBox:
        x, y, w, h = box
        return BBox(x / self.wW, y / self.wH, w / self.wW, h / self.wH).clamp()

    def _from_norm(self, bb: BBox):
        return _clip_box((bb.x * self.wW, bb.y * self.wH, bb.w * self.wW, bb.h * self.wH), self.wW, self.wH)

    def _detect_snap(self, work: np.ndarray, app: Appearance, box, min_iou: float = 0.25):
        """Snap the box to the best-overlapping detection of the same class."""
        if not self.detector or not app.label:
            return None
        cur = self._to_norm(box)
        best, best_iou = None, min_iou
        for d in self.detector.detect(work):
            if d.label != app.label:
                continue
            iou = cur.iou(d.bbox)
            if iou > best_iou:
                b = self._from_norm(d.bbox)
                if b is not None:
                    best, best_iou = b, iou
        return best

    def _local_correct(self, gray: np.ndarray, app: Appearance, box):
        """Multi-scale template match around the current box (scale/drift fix)."""
        x, y, w, h = box
        mx, my = int(w * 0.6), int(h * 0.6)
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(self.wW, x + w + mx), min(self.wH, y + h + my)
        roi = gray[y0:y1, x0:x1]
        if roi.shape[0] < 8 or roi.shape[1] < 8:
            return None
        hit = _match_scales(roi, app, LOCAL_SCALES, offset=(x0, y0))
        if hit and hit[1] >= LOCAL_CORRECT_MIN:
            return _clip_box(hit[0], self.wW, self.wH)
        return None

    def _search(self, work: np.ndarray, app: Appearance, use_detector: bool, use_template: bool):
        if use_detector and self.detector and app.label:
            cands = []
            for d in self.detector.detect(work):
                if d.label != app.label:
                    continue
                b = self._from_norm(d.bbox)
                if b is None:
                    continue
                ratio = (b[2] * b[3]) / max(1, app.size[0] * app.size[1])
                if 0.16 <= ratio <= 6.25:
                    cands.append((b, app.distance(work, b)))
            if cands:
                cands.sort(key=lambda c: c[1])
                if cands[0][1] < HIST_ACCEPT_THR + 0.12:
                    return cands[0][0]
        if use_template:
            gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
            hit = _match_scales(gray, app, SEARCH_SCALES)
            if hit and hit[1] >= TEMPLATE_ACCEPT:
                b = _clip_box(hit[0], self.wW, self.wH)
                if b and app.distance(work, b) < HIST_ACCEPT_THR:
                    return b
        return None

    # ---- frame producers -------------------------------------------------
    def _forward(self, start: int, last: int, first_frame: np.ndarray) -> Iterator[tuple[int, np.ndarray]]:
        step = max(1, self.o.step)
        yield start, first_frame
        idx = start + 1
        self.reader.seek(idx)
        while idx <= last:
            target = start + ((idx - start + step - 1) // step) * step
            while idx < target and idx <= last:
                if not self.reader.cap.grab():
                    return
                self.reader._pos += 1
                idx += 1
            if idx > last:
                return
            f = self.reader.read()
            if f is None:
                return
            yield idx, f
            idx += 1

    def _backward(self, start: int, first_frame: np.ndarray) -> Iterator[tuple[int, np.ndarray]]:
        step = max(1, self.o.step)
        yield start, first_frame
        CH = 64
        hi = start
        while hi > 0:
            lo = max(0, hi - CH)
            chunk = [(i, f) for i, f in self.reader.iter_frames(lo, hi) if (start - i) % step == 0]
            for i, f in reversed(chunk):
                yield i, f
            hi = lo

    # ---- core loop -------------------------------------------------------
    def _run(self, frames: Iterator[tuple[int, np.ndarray]], app: Appearance, init_box, spans: list[Span],
             progress: Optional[ProgressCB], cancel: Optional[CancelCB], total: int, label: str,
             lost_limit_frames: Optional[int] = None) -> int:
        o = self.o
        tracker = None
        state = "init"
        bad = 0
        n = 0
        lost_since = 0
        cur: Optional[Span] = None
        recent: list[int] = []
        for idx, frame in frames:
            if cancel and cancel():
                break
            n += 1
            if progress and n % 20 == 0:
                progress(min(0.999, n / max(1, total)), f"{label}: frame {idx}")
            if n % UPDATE_EVERY == 0:
                self._emit_update()
            work = self._work(frame)
            if state == "init":
                tracker = _make_tracker(o.tracker)
                tracker.init(work, tuple(int(v) for v in init_box))
                cur = Span(idx, idx, {idx: self._to_norm(init_box)})
                spans.append(cur)
                recent = [idx]
                state = "tracking"
                continue
            if state == "tracking":
                ok, box = tracker.update(work)
                if ok and o.tracker == "vit" and tracker_score(tracker) < VIT_MIN_SCORE:
                    ok = False
                b = _clip_box(box, self.wW, self.wH) if ok else None
                if b is not None and (n % o.correct_every == 0 or n % o.detect_every == 0):
                    nb = None
                    if n % o.detect_every == 0:
                        nb = self._detect_snap(work, app, b)
                    # the template correction is for scale on trackers that cannot adapt (KCF)
                    if nb is None and o.tracker != "vit" and n % o.correct_every == 0:
                        nb = self._local_correct(cv2.cvtColor(work, cv2.COLOR_BGR2GRAY), app, b)
                    if nb is not None and nb != b and self._to_norm(nb).iou(self._to_norm(b)) < REINIT_IOU:
                        b = nb
                        tracker = _make_tracker(o.tracker)
                        tracker.init(work, b)
                drift = app.distance(work, b)
                if b is None or drift > HIST_DRIFT_THR:
                    bad += 1
                else:
                    bad = 0
                if bad >= LOST_AFTER_BAD or (b is None and bad >= 2):
                    for k in recent[-bad:]:
                        cur.keyframes.pop(k, None)
                    keys = sorted(cur.keyframes)
                    if keys:
                        cur.start_frame, cur.end_frame = keys[0], keys[-1]
                    else:
                        spans.remove(cur)
                    state = "lost"
                    lost_since = n
                    bad = 0
                    recent = []
                    continue
                if b is not None:
                    cur.keyframes[idx] = self._to_norm(b)
                    cur.start_frame = min(cur.start_frame, idx)
                    cur.end_frame = max(cur.end_frame, idx)
                    recent.append(idx)
                    if len(recent) > 16:
                        recent = recent[-16:]
                continue
            # lost
            if lost_limit_frames is not None and (n - lost_since) * o.step > lost_limit_frames:
                break
            k = n - lost_since
            b = self._search(work, app, use_detector=(k % o.lost_detect_every == 0),
                             use_template=(k % o.lost_search_every == 0))
            if b is not None:
                tracker = _make_tracker(o.tracker)
                tracker.init(work, b)
                cur = Span(idx, idx, {idx: self._to_norm(b)})
                spans.append(cur)
                recent = [idx]
                state = "tracking"
                bad = 0
        return n

    # ---- public ----------------------------------------------------------
    def track(self, start_frame: int, bbox: BBox, label: Optional[str] = None,
              progress: Optional[ProgressCB] = None, cancel: Optional[CancelCB] = None,
              on_update: Optional[UpdateCB] = None) -> VideoTrack:
        o = self.o
        # anchored: objects with a class detector (faces, phones, screens, documents, plates) go to the
        # detector-anchored engine, which covered 96.7 % of confident faces against 86.1 % here; objects
        # without one (tattoos, notepads, "other") stay on this loop, which measured the same coverage as
        # the anchored no-detector path with a quarter of its drift (bench/results/TRACKING_V2.md).
        # anchored-all: the anchored engine for everything (its colour search re-acquires a flat object
        # that left the frame; this loop does not).
        use_anchored = o.engine == "anchored-all" or (o.engine == "anchored" and self.detector is not None and label is not None)
        if use_anchored and type(self) is ObjectTracker:
            from .anchored import AnchoredTracker
            identity = None
            if label == "face" and self.detector is not None:
                from .identity import FaceIdentity
                identity = FaceIdentity.try_create()
            return AnchoredTracker(self.reader, self.detector, o, identity=identity).track(
                start_frame, bbox, label=label, progress=progress, cancel=cancel, on_update=on_update)
        self._on_update = on_update
        reader = self.reader
        last = o.end_frame if o.end_frame is not None else max(0, reader.frame_count - 1)
        first = reader.read_at(start_frame)
        if first is None:
            raise RuntimeError(f"Cannot read frame {start_frame}")
        self._adapt_work_side(bbox)
        work0 = self._work(first)
        init_box = self._from_norm(bbox)
        if init_box is None:
            raise ValueError("Box too small")
        det_label = label if (label in DETECTOR_LABELS and self.detector is not None) else None
        app = Appearance.build(work0, init_box, det_label)
        step = max(1, o.step)
        total_fwd = max(0, last - start_frame) // step + 1
        total_bwd = (start_frame // step + 1) if o.backward else 0
        total = total_fwd + total_bwd
        spans: list[Span] = []

        fwd: list[Span] = []
        self._live = [fwd]
        limit = int(o.lost_limit_s * reader.fps)
        self._run(self._forward(start_frame, last, first), app, init_box, fwd, progress, cancel, total,
                  "Tracking forward", lost_limit_frames=None if o.search_to_end else limit)
        spans += fwd

        if o.backward and start_frame > 0 and not (cancel and cancel()):
            bwd: list[Span] = []
            self._live = [fwd, bwd]
            self._run(self._backward(start_frame, first), app, init_box, bwd,
                      progress, cancel, total, "Tracking backward", lost_limit_frames=limit)
            spans += bwd

        track = VideoTrack(label=label or "object", spans=[], source="tracked")
        for s in spans:
            keys = sorted(s.keyframes)
            if not keys:
                continue
            s.start_frame, s.end_frame = keys[0], keys[-1]
            track.spans.append(s)
        track.merge_spans()
        hold = max(o.pad_frames, o.hold_frames)
        for s in track.spans:
            s.start_frame = max(0, s.start_frame - hold)
            s.end_frame = min(last, s.end_frame + step - 1 + hold)
        track.merge_spans()
        bridge_spans(track, int(o.bridge_gap_s * reader.fps))
        if progress:
            progress(1.0, "Tracking complete")
        return track


def bridge_spans(track: VideoTrack, max_gap: int, near=None) -> VideoTrack:
    """Join consecutive spans of one tracked object whose gap is <= max_gap frames and whose
    boundary boxes are near each other (same object, briefly lost); the gap is then covered by
    interpolation between the two boundary keyframes. In place; returns the track.
    `near(a, b, gap)` overrides the default nearness test."""
    if max_gap <= 0 or len(track.spans) < 2:
        return track
    spans = sorted(track.spans, key=lambda s: s.start_frame)
    out = [spans[0]]
    for s in spans[1:]:
        prev = out[-1]
        gap = s.start_frame - prev.end_frame
        a, b = prev.bbox_at(prev.end_frame), s.bbox_at(s.start_frame)
        close = (near(a, b, gap) if near else _near_boxes(a, b)) if (a is not None and b is not None) else False
        if 0 < gap <= max_gap and close:
            prev.keyframes.update(s.keyframes)
            prev.end_frame = max(prev.end_frame, s.end_frame)
        else:
            out.append(s)
    track.spans = out
    track.merge_spans()
    return track


def _near_boxes(a: BBox, b: BBox) -> bool:
    """Similar size and centres within ~1.5 box-sizes: plausibly the same object."""
    sa, sb = max(a.w, a.h), max(b.w, b.h)
    if sa <= 0 or sb <= 0 or not (0.5 <= sa / sb <= 2.0):
        return False
    (ax, ay), (bx, by) = a.center(), b.center()
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 <= 1.5 * max(sa, sb)
