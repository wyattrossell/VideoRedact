"""Anchored single-object tracker (engine "anchored", 2026-10-29; Phase 1 of the tracking rebuild).

A manually drawn box is a statement that this object must be covered for its whole visible life
around the drawn frame. The first engine (`ObjectTracker` in tracker.py) worked on a <=512 px
copy of the frame, compared every frame with the original patch and gave up when ViTTrack's
score dipped; measured on 20 confident faces (bench/track_eval.py) it covered 87 % of their
frames with the face detector and 72 % without. This engine:

  * works at full resolution: ViTTrack crops its own search window so its cost does not depend
    on the frame size; the class detector runs on a region around the predicted box (a 400 px
    YuNet crop costs ~8 ms) and, while the object is lost, on the whole frame every few
    processed frames;
  * keeps an appearance bank of several confident views (grey template + colour histogram, and
    an SFace identity embedding for faces when the model is installed) instead of the original
    patch, so a head that turned or a plate that caught the headlights can still be recognised;
  * declares "lost" with hysteresis: the detector not seeing the object for a while *and* the
    tracker unsure, the tracker unsure for longer, no detector confirmation for a few seconds,
    or the box leaving the frame;
  * re-acquires with gates: same class, plausible size and position for the time lost, and an
    identity / appearance match to the bank;
  * trims the uncertain tail before a loss, holds the box briefly after it, bridges short gaps
    and runs the same loop backward from the drawn frame (all inherited from ObjectTracker).
"""
from __future__ import annotations

from typing import Iterator, Optional

import cv2
import numpy as np

from videoredact.core.model import BBox, Span, VideoTrack
from .detectors.base import Detection
from .tracker import (CancelCB, ObjectTracker, ProgressCB, UPDATE_EVERY, UpdateCB, _clip_box, _hist,
                      bridge_spans, make_tracker, release_tracker, tracker_score)

TPL = 48                  # side of the grey templates in the appearance bank
BANK_MAX = 8              # views kept (the user's box is always kept)
BANK_EVERY = 10           # processed frames between bank additions while confirmed
DET_EVERY = 5             # processed frames between detector checks while tracking
S_WEAK = 0.35             # ViT score below this: unsure on this frame
S_GOOD = 0.55             # ViT score at or above this: confident
MISS_LOST = 3             # detector checks without a match while unsure -> lost
WEAK_LOST = 6             # consecutive unsure frames without a detector -> lost
WEAK_LOST_DET = 12        # consecutive unsure frames even with a detector -> lost
UNCONFIRMED_MAX_S = 3.0   # faces: seconds without a detector confirmation -> lost (even if ViT is confident)
UNCONFIRMED_MAX_S_OBJ = 8.0  # other detector classes (sporadic detectors): the same limit
APP_MIN = 0.45            # template correlation against the bank below this counts against the object
HIST_MAX = 0.45           # Bhattacharyya distance above this: colours do not match
HIST_STRONG = 0.25        # ... and below this the colours alone say it is the object
HIST_LOST = 0.70          # ... and above this (while tracking without a detector) the colours say it is gone
WEAK_APP_LOST = 6         # consecutive frames the appearance counts against the object -> lost (no detector)
ROI_FACTOR = 4.0          # detector region = this many box sides around the prediction
ROI_MIN = 256
IDENT_MIN = 0.363         # SFace cosine for "same person" (OpenCV's documented threshold)
TEMPLATE_ACCEPT = 0.55    # template search correlation needed to re-acquire without a detector (colours must match too)
TEMPLATE_ACCEPT_LATE = 0.68  # ... when the search has gone frame-wide
SEARCH_SCALES = (0.75, 0.9, 1.0, 1.15, 1.35)
REINIT_IOU = 0.85         # re-create the ViT tracker only when a detector snap moved the box this much
TRIM_MAX = 16             # most uncertain keyframes dropped before a loss
INSIDE_MIN = 0.3          # share of the box that must remain inside the frame
TPL_MIN_STD = 4.0         # grey-level spread below which a template is "flat" and not correlated
TRACK_DET_THR = 0.30      # detector score while tracking a known object (the ROI, size and identity gates do the rest)
UPSCALE_BELOW = 120.0     # objects expected smaller than this (px) are upscaled before detection ...
UPSCALE_MAX = 2.0         # ... by at most this factor ...
UPSCALE_MAX_SIDE = 1920   # ... and never beyond this crop side
S_OK = 0.45               # ViT score good enough to vouch for a frame when nothing else can (no detector)
IDENT_SOFT = 0.25         # identity needed for a candidate close to where the face was lost (looks must match too)

PxBox = tuple[int, int, int, int]


def _norm(b: PxBox, W: int, H: int) -> BBox:
    x, y, w, h = b
    return BBox(x / W, y / H, w / W, h / H).clamp()


def _iou(a: PxBox, b: PxBox) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _center(b: PxBox) -> tuple[float, float]:
    return b[0] + b[2] / 2.0, b[1] + b[3] / 2.0


def _side(b: PxBox) -> float:
    return float(max(b[2], b[3]))


def _roi(W: int, H: int, box: PxBox, factor: float = ROI_FACTOR, min_side: int = ROI_MIN) -> tuple[int, int, int, int]:
    """Square region (x0, y0, x1, y1) of `factor` box-sides around the box, clamped to the frame."""
    side = int(max(min_side, factor * _side(box)))
    side = min(side, max(W, H))
    cx, cy = _center(box)
    x0 = int(max(0, min(W - side, cx - side / 2)))
    y0 = int(max(0, min(H - side, cy - side / 2)))
    return x0, y0, min(W, x0 + side), min(H, y0 + side)


def _detect_in(det, frame: np.ndarray, region: tuple[int, int, int, int], label: str,
               expect: float = 0.0) -> list[tuple[PxBox, Detection]]:
    """Run the class detector on a region; boxes (pixels) and detections come back in frame
    coordinates. `expect` is the object's expected side in pixels: small objects are upscaled
    before detection (YuNet misses the faces auto-detect only finds on its 1.5x pass)."""
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = region
    crop = frame[y0:y1, x0:x1]
    if crop.shape[0] < 16 or crop.shape[1] < 16:
        return []
    up = 1.0
    if expect > 0 and expect < UPSCALE_BELOW:
        up = min(UPSCALE_MAX, UPSCALE_BELOW / expect)
        if max(crop.shape[:2]) * up > UPSCALE_MAX_SIDE:
            up = max(1.0, UPSCALE_MAX_SIDE / max(crop.shape[:2]))
        if up > 1.01:
            crop = cv2.resize(crop, (int(crop.shape[1] * up), int(crop.shape[0] * up)), interpolation=cv2.INTER_LINEAR)
    cw, ch = x1 - x0, y1 - y0
    out = []
    for d in det.detect(crop):
        if d.label != label:
            continue
        b = d.bbox
        px = _clip_box((b.x * cw + x0, b.y * ch + y0, b.w * cw, b.h * ch), W, H)
        if px is None:
            continue
        lm = None
        if getattr(d, "landmarks", None) is not None:
            lm = np.asarray(d.landmarks, dtype=np.float32).reshape(5, 2).copy()
            lm[:, 0] = (lm[:, 0] * cw + x0) / W
            lm[:, 1] = (lm[:, 1] * ch + y0) / H
        out.append((px, Detection(_norm(px, W, H), d.label, float(d.score), getattr(d, "source", ""), lm)))
    return out


def _sane(b: PxBox, ref: PxBox, area=(0.3, 3.0), aspect=(0.55, 1.8)) -> bool:
    """A tracker box is believable while its area and aspect stay within the given ratios of a
    reference box: the last detector-confirmed box (ViT otherwise balloons over a whole car in a
    few frames), or, without a detector, just the previous box with tight per-step limits, so a
    sudden jump is caught but gradual change (a head turning, a car approaching) is allowed."""
    a, r = float(b[2] * b[3]), float(max(1, ref[2] * ref[3]))
    if not (area[0] <= a / r <= area[1]):
        return False
    asp = (b[2] / max(1.0, b[3])) / (ref[2] / max(1.0, ref[3]))
    return aspect[0] <= asp <= aspect[1]


def plausible(last: PxBox, cand: PxBox, gap_frames: int, vel: tuple[float, float], late: bool) -> bool:
    """Could `cand` be the object last seen at `last` `gap_frames` ago? Size within a factor
    (wider after long gaps), and, until the search goes frame-wide, within the distance the
    object could have moved (its recent speed, with a floor of 2.5 box sides)."""
    sa, sb = _side(last), _side(cand)
    if sa <= 0 or sb <= 0:
        return False
    r = sb / sa
    if not ((0.35 <= r <= 3.0) if late else (0.5 <= r <= 2.0)):
        return False
    if late:
        return True
    (ax, ay), (bx, by) = _center(last), _center(cand)
    speed = (vel[0] ** 2 + vel[1] ** 2) ** 0.5
    reach = max(2.5 * sa, speed * gap_frames + 2.0 * sa)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 <= reach


class Bank:
    """Appearance bank: grey templates, colour histograms and (faces) identity embeddings of
    confident views. Index 0 is the user's own box and is never dropped."""

    def __init__(self, identity=None):
        self.templates: list[np.ndarray] = []
        self.hists: list[np.ndarray] = []
        self.sizes: list[tuple[int, int]] = []
        self.feats: list[np.ndarray] = []
        self.identity = identity

    @property
    def textured(self) -> list[np.ndarray]:
        """Templates with enough contrast for a normalized correlation to mean anything (a flat
        colour patch correlates with anything or with nothing, depending on rounding)."""
        return [t for t in self.templates if float(t.std()) >= TPL_MIN_STD]

    @staticmethod
    def _inner(frame: np.ndarray, box: PxBox) -> Optional[np.ndarray]:
        """The inner 70 % of the box: a tracker box is loose and grows, and the strip of background
        it includes must not decide whether the object still looks like itself."""
        x, y, w, h = box
        dx, dy = int(w * 0.15), int(h * 0.15)
        p = frame[y + dy:y + h - dy, x + dx:x + w - dx]
        if p.size == 0 or p.shape[0] < 4 or p.shape[1] < 4:
            return None
        return p

    @classmethod
    def _grey(cls, frame: np.ndarray, box: PxBox) -> Optional[np.ndarray]:
        p = cls._inner(frame, box)
        if p is None:
            return None
        return cv2.resize(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY), (TPL, TPL), interpolation=cv2.INTER_AREA)

    def add(self, frame: np.ndarray, box: PxBox, det: Optional[Detection] = None, force: bool = False) -> None:
        g = self._grey(frame, box)
        if g is None:
            return
        same = self.ncc(g) if (not force and self.templates) else None
        if same is not None and same > 0.92:
            pass                                   # nothing new to learn from this view
        else:
            self.templates.append(g)
            self.hists.append(_hist(self._inner(frame, box)))
            self.sizes.append((box[2], box[3]))
            if len(self.templates) > BANK_MAX:
                del self.templates[1], self.hists[1], self.sizes[1]
        if self.identity is not None and det is not None and det.landmarks is not None:
            f = self.identity.embed(frame, det)
            if f is not None:
                self.feats.append(f)
                if len(self.feats) > BANK_MAX:
                    del self.feats[1]

    def ncc(self, g: np.ndarray) -> Optional[float]:
        """Best correlation of a 48x48 grey patch with the textured templates; None when the
        comparison is meaningless (no textured template, or a flat patch)."""
        if float(g.std()) < TPL_MIN_STD:
            return None
        best = None
        for t in self.textured:
            v = float(cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED)[0, 0])
            if np.isfinite(v) and (best is None or v > best):
                best = v
        return best

    def appearance(self, frame: np.ndarray, box: PxBox) -> tuple[Optional[float], float]:
        """(best template correlation or None, smallest histogram distance) of the box against the bank."""
        g = self._grey(frame, box)
        if g is None or not self.templates:
            return None, 1.0
        h_ = _hist(self._inner(frame, box))
        hd = min(float(cv2.compareHist(hb, h_, cv2.HISTCMP_BHATTACHARYYA)) for hb in self.hists)
        return self.ncc(g), hd

    def looks_like(self, frame: np.ndarray, box: PxBox) -> bool:
        """Colours must match; the grey pattern too whenever a correlation can be computed,
        unless the colour match is strong enough on its own."""
        ncc, hd = self.appearance(frame, box)
        return hd <= HIST_MAX and (ncc is None or ncc >= APP_MIN or hd <= HIST_STRONG)

    def identity_score(self, frame: np.ndarray, det: Detection) -> Optional[float]:
        """Best cosine similarity to the bank's identities, or None when identity cannot be used."""
        if self.identity is None or not self.feats or det.landmarks is None:
            return None
        f = self.identity.embed(frame, det)
        if f is None:
            return None
        return max(float(np.dot(f, g)) for g in self.feats)


class AnchoredTracker(ObjectTracker):
    """Same interface as ObjectTracker.track(); see the module docstring for the differences."""

    def __init__(self, reader, detector=None, opts=None, identity=None):
        super().__init__(reader, detector, opts)
        self.identity = identity
        self.label: Optional[str] = None
        self.det = detector
        if detector is not None and hasattr(detector, "set_score_threshold"):
            detector.set_score_threshold(TRACK_DET_THR)
        self.debug = None        # optional callable(dict) receiving the per-frame decision variables

    # ---- helpers ---------------------------------------------------------
    def _snap(self, frame: np.ndarray, b: PxBox) -> Optional[tuple[PxBox, Detection]]:
        """Best same-class detection around the predicted box: overlapping it, or centred within
        one box side of it (ViT boxes drift and shrink; a detector box that is close is the object)."""
        H, W = frame.shape[:2]
        best, best_v = None, 0.0
        for nb, d in _detect_in(self.det, frame, _roi(W, H, b), self.label, expect=_side(b)):
            v = _iou(nb, b)
            if v < 0.2:
                (ax, ay), (bx, by) = _center(b), _center(nb)
                if ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 > _side(b) or not (0.5 <= _side(nb) / _side(b) <= 2.0):
                    continue
                v = 0.2
            v *= float(d.score)
            if v > best_v:
                best, best_v = (nb, d), v
        return best

    def _template_search(self, frame: np.ndarray, bank: Bank, last: PxBox, region: tuple[int, int, int, int],
                         accept: float = TEMPLATE_ACCEPT) -> Optional[PxBox]:
        """Multi-scale search for the bank's most recent templates inside `region`, on a half-size
        grey copy when the frame is wide. Accepts a correlation >= `accept` whose colours
        match the bank."""
        H, W = frame.shape[:2]
        x0, y0, x1, y1 = region
        sub = frame[y0:y1, x0:x1]
        if sub.shape[0] < 16 or sub.shape[1] < 16:
            return None
        s = 0.5 if max(sub.shape[:2]) > 360 else 1.0     # half-res smooths small differences (and halves the cost)
        grey = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
        if s < 1.0:
            grey = cv2.resize(grey, (int(grey.shape[1] * s), int(grey.shape[0] * s)), interpolation=cv2.INTER_AREA)
        gh, gw = grey.shape
        lw, lh = last[2], last[3]
        best = None
        for t in bank.textured[-3:]:
            for sc in SEARCH_SCALES:
                w, h = int(round(lw * s * sc)), int(round(lh * s * sc))
                if w < 8 or h < 8 or w >= gw or h >= gh:
                    continue
                tpl = cv2.resize(t, (w, h), interpolation=cv2.INTER_LINEAR)
                res = cv2.matchTemplate(grey, tpl, cv2.TM_CCOEFF_NORMED)
                _, mx, _, loc = cv2.minMaxLoc(res)
                if np.isfinite(mx) and (best is None or mx > best[1]):
                    best = ((loc[0] / s + x0, loc[1] / s + y0, w / s, h / s), float(mx))
        if best is None or best[1] < accept:
            return None
        b = _clip_box(best[0], W, H)
        if b is None or not bank.looks_like(frame, b):
            return None
        return b

    def _colour_search(self, frame: np.ndarray, bank: Bank, last: PxBox, region: tuple[int, int, int, int]) -> Optional[PxBox]:
        """Search for a flat-coloured object (no usable grey texture, e.g. a plain notepad) by
        back-projecting the bank's colour histograms inside `region` and taking the best blob of
        a plausible size."""
        H, W = frame.shape[:2]
        x0, y0, x1, y1 = region
        sub = frame[y0:y1, x0:x1]
        if sub.shape[0] < 16 or sub.shape[1] < 16 or not bank.hists:
            return None
        hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
        acc = None
        for h in bank.hists[-3:]:
            bp = cv2.calcBackProject([hsv], [0, 1], h, [0, 180, 0, 256], 1)
            acc = bp if acc is None else np.maximum(acc, bp)
        mask = (acc > 0.5).astype(np.uint8)
        k = max(3, int(min(last[2], last[3]) * 0.1) | 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        area0 = float(last[2] * last[3])
        best, best_v = None, 0.0
        for i in range(1, n):
            x, y, w, h, a = (int(v) for v in stats[i])
            if not (0.25 * area0 <= w * h <= 4.0 * area0) or not (0.5 <= (w / max(1, h)) / (last[2] / max(1, last[3])) <= 2.0):
                continue
            fill = a / float(w * h)
            if fill < 0.5:
                continue
            if a * fill > best_v:
                best, best_v = (x + x0, y + y0, w, h), a * fill
        if best is None:
            return None
        b = _clip_box(best, W, H)
        if b is None or not bank.looks_like(frame, b):
            return None
        return b

    # ---- core loop -------------------------------------------------------
    def _run(self, frames: Iterator[tuple[int, np.ndarray]], bank: Bank, init_box: PxBox, spans: list[Span],
             progress: Optional[ProgressCB], cancel: Optional[CancelCB], total: int, label: str,
             lost_limit_frames: Optional[int] = None) -> int:
        o = self.o
        fps = self.reader.fps or 30.0
        # faces have a strong detector and an identity check, so a face the detector stops seeing is
        # given up quickly; object detectors (phones at 0.35-0.6) fire sporadically, so those tracks
        # live longer on the tracker alone and may re-acquire by appearance as well
        strict = self.label == "face"
        unconfirmed_max = max(1, int((UNCONFIRMED_MAX_S if strict else UNCONFIRMED_MAX_S_OBJ) * fps / max(1, o.step)))
        tracker = None
        state = "init"
        cur: Optional[Span] = None
        n = 0
        weak = det_miss = since_confirm = weak_app = lost_since = 0
        recent: list[tuple[int, bool]] = []        # (frame, confident?) for trimming before a loss
        last_box: Optional[PxBox] = None
        last_idx: Optional[int] = None
        ref_box: PxBox = init_box                   # last box somebody vouched for (detector / confident + looks)
        vel = (0.0, 0.0)
        for idx, frame in frames:
            if cancel and cancel():
                break
            n += 1
            if progress and n % 20 == 0:
                progress(min(0.999, n / max(1, total)), f"{label}: frame {idx}")
            if n % UPDATE_EVERY == 0:
                self._emit_update()
            H, W = frame.shape[:2]
            if state == "init":
                tracker = make_tracker(o.tracker)
                tracker.init(frame, tuple(int(v) for v in init_box))
                cur = Span(idx, idx, {idx: _norm(init_box, W, H)})
                spans.append(cur)
                recent = [(idx, True)]
                last_box, last_idx = init_box, idx
                state = "tracking"
                continue
            if state == "tracking":
                ok, raw = tracker.update(frame)
                s = tracker_score(tracker) if ok else 0.0
                b = _clip_box(raw, W, H) if ok else None
                inside = sane = True
                if b is not None:
                    inside = (b[2] * b[3]) >= INSIDE_MIN * max(1.0, float(raw[2]) * float(raw[3]))
                    # ViT can balloon or collapse a box in a few frames: against the last confirmed
                    # box with a detector, against the previous box (per-step limits) without one
                    sane = _sane(b, ref_box) if self.det is not None else \
                        _sane(b, last_box or b, area=(0.5, 2.0), aspect=(0.67, 1.5))
                confirmed = False
                # the detector looks on a schedule, and right away whenever the tracker is unsure,
                # failed, or produced an implausible box (a brief ViT failure then costs nothing)
                if self.det is not None and (n % DET_EVERY == 0 or s < S_WEAK or b is None or not sane):
                    hit = self._snap(frame, b if (b is not None and sane) else last_box)
                    if hit is not None:
                        nb, d = hit
                        confirmed = True
                        det_miss = 0
                        since_confirm = 0
                        if b is None or _iou(nb, b) < REINIT_IOU:
                            release_tracker(tracker, o.tracker)
                            tracker = make_tracker(o.tracker)
                            tracker.init(frame, nb)
                        b, sane, inside = nb, True, True
                        ref_box = nb
                        if n % BANK_EVERY == 0 or len(bank.templates) < 2:
                            bank.add(frame, b, d)
                    else:
                        det_miss += 1
                if not confirmed:
                    since_confirm += 1
                if b is not None and not sane and self.det is not None:
                    b = None                        # not a box worth recording; counts as a failed frame below
                                                    # (without a detector the appearance checks judge it instead)
                # Appearance check against the bank: every frame without a detector (ViT stays
                # confident on background texture once the object has left), and once the detector
                # has missed a few times with one.
                looks = True
                hd_now = 0.0
                if b is not None and (self.det is None or det_miss >= MISS_LOST):
                    ncc, hd_now = bank.appearance(frame, b)
                    looks = hd_now <= HIST_MAX and (ncc is None or ncc >= APP_MIN or hd_now <= HIST_STRONG)
                if self.det is None:
                    # the appearance is the only witness: a frame counts against the object when its
                    # colours are clearly wrong and the tracker is not confident either
                    bad_app = b is not None and hd_now > HIST_LOST and s < S_GOOD
                else:
                    bad_app = b is not None and not looks
                weak_app = weak_app + 1 if bad_app else 0
                if self.det is None and b is not None and s >= S_OK and looks:
                    ref_box = b
                    if n % BANK_EVERY == 0:
                        bank.add(frame, b)
                weak = weak + 1 if (b is None or s < S_WEAK or not sane) else 0
                lost = (b is None and weak >= 2) or (b is not None and not inside)
                if self.det is not None:
                    lost = lost or (det_miss >= MISS_LOST and (weak >= 1 or not looks)) \
                        or (strict and det_miss >= 2 * MISS_LOST) \
                        or since_confirm >= unconfirmed_max or weak >= WEAK_LOST_DET
                else:
                    lost = lost or weak >= WEAK_LOST or weak_app >= WEAK_APP_LOST
                if self.debug:
                    self.debug(dict(idx=idx, score=round(s, 2), box=b, confirmed=confirmed, det_miss=det_miss,
                                    since_confirm=since_confirm, weak=weak, weak_app=weak_app, looks=looks,
                                    inside=inside, lost=lost, bank=len(bank.templates)))
                if lost:
                    # drop the terminal stretch that nothing confirmed (all of it with a detector:
                    # the hold after the last confirmed frame covers a fade-out; a bounded tail without)
                    cap = 64 if self.det is not None else TRIM_MAX
                    trim = 0
                    for _, good in reversed(recent):
                        if good or trim >= cap:
                            break
                        trim += 1
                    for fidx, _ in recent[len(recent) - trim:] if trim else []:
                        cur.keyframes.pop(fidx, None)
                    keys = sorted(cur.keyframes)
                    if keys:
                        cur.start_frame, cur.end_frame = keys[0], keys[-1]
                    else:
                        spans.remove(cur)
                    release_tracker(tracker, o.tracker)
                    tracker = None
                    state = "lost"
                    lost_since = n
                    weak = det_miss = since_confirm = weak_app = 0
                    recent = []
                    continue
                if b is not None:
                    cur.keyframes[idx] = _norm(b, W, H)
                    cur.start_frame = min(cur.start_frame, idx)
                    cur.end_frame = max(cur.end_frame, idx)
                    recent.append((idx, confirmed if self.det is not None else (s >= S_OK and looks and sane)))
                    if len(recent) > 96:
                        recent = recent[-96:]
                    if last_box is not None and last_idx is not None and idx != last_idx:
                        dt = float(idx - last_idx)
                        (cx, cy), (lx, ly) = _center(b), _center(last_box)
                        vel = (0.7 * vel[0] + 0.3 * (cx - lx) / dt, 0.7 * vel[1] + 0.3 * (cy - ly) / dt)
                    last_box, last_idx = b, idx
                continue
            # lost: look for the object again
            k = n - lost_since
            if lost_limit_frames is not None and k * o.step > lost_limit_frames:
                break
            gap = abs(idx - (last_idx if last_idx is not None else idx))
            late = gap > fps * 1.0
            region = (0, 0, W, H) if late else _roi(W, H, last_box, factor=2.0 + 0.5 * k, min_side=ROI_MIN)
            cand: Optional[PxBox] = None
            rejected: list = []
            if self.det is not None and k % o.lost_detect_every == 0:
                best_v = 0.0
                for nb, d in _detect_in(self.det, frame, region, self.label, expect=_side(ref_box)):
                    if not plausible(ref_box, nb, gap, vel, late):
                        rejected.append((nb, "implausible"))
                        continue
                    sim = bank.identity_score(frame, d)
                    if sim is not None:
                        # a strong identity match anywhere; a weaker one only close to where the face
                        # was lost and only when it also looks like the bank (profile views score low)
                        if sim < IDENT_MIN and not (sim >= IDENT_SOFT and not late and bank.looks_like(frame, nb)):
                            rejected.append((nb, f"identity {sim:.2f}"))
                            continue
                        v = sim
                    else:
                        ncc, hd = bank.appearance(frame, nb)
                        if (ncc is not None and ncc < APP_MIN) and hd > HIST_MAX:
                            rejected.append((nb, f"appearance ncc {ncc:.2f} hist {hd:.2f}"))
                            continue
                        v = max(ncc if ncc is not None else 0.0, 1.0 - hd)
                    v *= float(d.score)
                    if v > best_v:
                        best_v, cand = v, nb
            if cand is None and not strict and k % o.lost_search_every == 0 and (self.det is None or k % o.lost_detect_every != 0):
                # appearance search for everything but faces (template matches on a face drifted into
                # garbage boxes on the reference clips; a face waits for its detector + identity check)
                if bank.textured:
                    # a frame-wide search has far more chances to hit lookalikes: demand more
                    cand = self._template_search(frame, bank, last_box, region,
                                                 accept=TEMPLATE_ACCEPT_LATE if late else TEMPLATE_ACCEPT)
                else:
                    cand = self._colour_search(frame, bank, last_box, region)
                if cand is not None and not plausible(last_box, cand, gap, vel, late):
                    cand = None
            if self.debug and (cand is not None or rejected):
                self.debug(dict(idx=idx, lost_k=k, late=late, cand=cand, rejected=rejected))
            if cand is not None:
                ref_box = cand
                tracker = make_tracker(o.tracker)
                tracker.init(frame, cand)
                cur = Span(idx, idx, {idx: _norm(cand, W, H)})
                spans.append(cur)
                recent = [(idx, True)]
                last_box, last_idx = cand, idx
                vel = (0.0, 0.0)
                state = "tracking"
                weak = det_miss = since_confirm = weak_app = 0
        if tracker is not None:
            release_tracker(tracker, o.tracker)
        return n

    # ---- public ----------------------------------------------------------
    def track(self, start_frame: int, bbox: BBox, label: Optional[str] = None,
              progress: Optional[ProgressCB] = None, cancel: Optional[CancelCB] = None,
              on_update: Optional[UpdateCB] = None) -> VideoTrack:
        o = self.o
        self._on_update = on_update
        reader = self.reader
        last = o.end_frame if o.end_frame is not None else max(0, reader.frame_count - 1)
        first = reader.read_at(start_frame)
        if first is None:
            raise RuntimeError(f"Cannot read frame {start_frame}")
        H, W = first.shape[:2]
        init_box = _clip_box((bbox.x * W, bbox.y * H, bbox.w * W, bbox.h * H), W, H)
        if init_box is None:
            raise ValueError("Box too small")
        self.label = label if (label is not None and self.det is not None) else None
        if self.label is None:
            self.det = None
        bank = Bank(self.identity if self.label == "face" else None)
        bank.add(first, init_box, force=True)
        if self.det is not None:                      # the user's box may be loose: learn the detector's view too
            hit = self._snap(first, init_box)
            if hit is not None:
                bank.add(first, hit[0], hit[1], force=True)
        step = max(1, o.step)
        total_fwd = max(0, last - start_frame) // step + 1
        total_bwd = (start_frame // step + 1) if o.backward else 0
        total = total_fwd + total_bwd
        limit = int(o.lost_limit_s * reader.fps)

        fwd: list[Span] = []
        self._live = [fwd]
        self._run(self._forward(start_frame, last, first), bank, init_box, fwd, progress, cancel, total,
                  "Tracking forward", lost_limit_frames=None if o.search_to_end else limit)
        spans = list(fwd)
        if o.backward and start_frame > 0 and not (cancel and cancel()):
            bwd: list[Span] = []
            self._live = [fwd, bwd]
            self._run(self._backward(start_frame, first), bank, init_box, bwd, progress, cancel, total,
                      "Tracking backward", lost_limit_frames=limit)
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
        fps = reader.fps or 30.0

        def near(a: BBox, b: BBox, gap: int) -> bool:
            # same object, briefly lost: similar size, and within a distance the object could
            # have moved. A body cam pans fast, so a gap under a second allows up to 4.5 sides;
            # longer gaps keep the strict 1.5 (an object that left and came back elsewhere must
            # not get a box sweeping across the frame).
            sa, sb = max(a.w, a.h), max(b.w, b.h)
            if sa <= 0 or sb <= 0 or not (0.5 <= sa / sb <= 2.0):
                return False
            (ax, ay), (bx, by) = a.center(), b.center()
            reach = 1.5 + 3.0 * gap / fps if gap <= fps else 1.5
            return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 <= max(sa, sb) * reach

        bridge_spans(track, int(o.bridge_gap_s * fps), near=near)
        if progress:
            progress(1.0, "Tracking complete")
        return track
