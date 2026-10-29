"""Anchored single-object tracker (vision/anchored.py): gating, appearance bank, and the
synthetic clip (a red square that leaves the frame at 4 s and returns at 6 s)."""
from pathlib import Path

import numpy as np
import pytest

from videoredact.core.media import FrameReader
from videoredact.core.model import BBox
from videoredact.vision import anchored as A
from videoredact.vision.tracker import ObjectTracker, TrackOptions

SAMPLE = Path("tests/data/sample.mp4")


def test_plausible_gates_size_and_distance():
    last = (100, 100, 50, 50)
    assert A.plausible(last, (130, 110, 50, 50), gap_frames=5, vel=(0.0, 0.0), late=False)
    assert not A.plausible(last, (600, 100, 50, 50), gap_frames=5, vel=(0.0, 0.0), late=False)   # too far for a slow object
    assert A.plausible(last, (600, 100, 50, 50), gap_frames=50, vel=(10.0, 0.0), late=False)     # fast object: reachable
    assert not A.plausible(last, (110, 100, 150, 150), gap_frames=5, vel=(0.0, 0.0), late=False) # 3x the size
    assert A.plausible(last, (900, 500, 120, 120), gap_frames=100, vel=(0.0, 0.0), late=True)    # frame-wide search
    assert not A.plausible(last, (900, 500, 10, 10), gap_frames=100, vel=(0.0, 0.0), late=True)  # still not 5x smaller


def _frame_with_patch(colour, texture=False, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    f[50:150, 50:150] = colour
    if texture:
        f[50:150, 50:150, 0] = rng.integers(0, 255, (100, 100), dtype=np.uint8)
    return f


def test_bank_flat_patch_uses_colour_only_and_rejects_background():
    f = _frame_with_patch((0, 0, 220))
    bank = A.Bank()
    bank.add(f, (50, 50, 100, 100), force=True)
    assert not bank.textured                                   # a flat colour has no grey texture to correlate
    assert bank.looks_like(f, (52, 48, 100, 100))              # same object, slightly shifted box
    assert not bank.looks_like(f, (0, 0, 60, 60))              # random background
    # learning a textured view afterwards must not make the flat object fail its own check
    g = _frame_with_patch((0, 0, 220), texture=True)
    bank.add(g, (50, 50, 100, 100), force=True)
    assert bank.textured
    assert bank.looks_like(f, (50, 50, 100, 100))


def test_bank_keeps_users_view_and_caps_size():
    bank = A.Bank()
    for i in range(A.BANK_MAX + 5):
        bank.add(_frame_with_patch((i * 10 % 255, 100, 50), texture=True, seed=i), (50, 50, 100, 100), force=True)
    assert len(bank.templates) == A.BANK_MAX
    first = _frame_with_patch((0, 100, 50), texture=True, seed=0)
    assert bank.looks_like(first, (50, 50, 100, 100))          # index 0 (the user's box) is never dropped


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample clip missing (run scripts/make_sample.py)")
def test_square_tracked_lost_offscreen_and_reacquired():
    with FrameReader(SAMPLE) as r:
        f = r.read_at(30)
        m = (f[:, :, 2] > 180) & (f[:, :, 1] < 60) & (f[:, :, 0] < 60)
        ys, xs = np.where(m)
        box = BBox(xs.min() / 640, ys.min() / 360, (xs.max() - xs.min()) / 640, (ys.max() - ys.min()) / 360)
        # "anchored-all": the anchored engine also for objects without a detector (the default
        # "anchored" routes those to the classic loop, which cannot re-acquire a flat object)
        tr = ObjectTracker(r, None, TrackOptions(engine="anchored-all", step=1)).track(30, box, label="other")
        wrong = missing = n = 0
        for fr in range(0, 360, 5):
            b = tr.bbox_at(fr)
            img = r.read_at(fr)
            m = (img[:, :, 2] > 180) & (img[:, :, 1] < 60) & (img[:, :, 0] < 60)
            ys, xs = np.where(m)
            real = BBox(xs.min() / 640, ys.min() / 360, (xs.max() - xs.min()) / 640, (ys.max() - ys.min()) / 360) if len(xs) > 50 else None
            if real is not None:
                n += 1
                if b is None:
                    missing += 1
                elif b.iou(real) < 0.3:
                    wrong += 1
            elif b is not None and 125 <= fr <= 165:
                wrong += 1                                     # a box while the square is off screen
    assert tr.source == "tracked"
    assert tr.bbox_at(60) is not None and tr.bbox_at(300) is not None   # before it left, and after it came back
    assert tr.bbox_at(145) is None                                       # nothing while it was gone
    assert missing <= 3 and wrong <= 2, (missing, wrong, n)


def test_classic_engine_still_selectable():
    o = TrackOptions(engine="classic")
    assert o.engine == "classic" and TrackOptions().engine == "anchored"


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample clip missing (run scripts/make_sample.py)")
def test_default_engine_routes_no_detector_objects_to_classic(monkeypatch):
    """With engine="anchored" (the default) an object without a detector uses the classic loop."""
    import videoredact.vision.anchored as anchored_mod
    called = []
    monkeypatch.setattr(anchored_mod.AnchoredTracker, "track", lambda self, *a, **k: called.append(1) or None)
    with FrameReader(SAMPLE) as r:
        ObjectTracker(r, None, TrackOptions(engine="anchored", step=4, end_frame=20, backward=False)).track(
            5, BBox(0.3, 0.3, 0.1, 0.1), label="other")
    assert not called
