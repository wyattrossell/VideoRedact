"""Ground truth from a hand-redacted reference copy.

Given an ORIGINAL recording and a REDACTED copy of it made by a person (any tool), diff the
two at regular timestamps: wherever the redacted copy differs from the original beyond
compression noise, a human decided something had to be covered. Those regions become
ground-truth boxes in the benchmark's label format (class "redacted"), so bench/run_bench.py
can measure leak rate / recall of our auto-detect on the original without anyone drawing labels.

    python bench/labels_from_reference.py ORIGINAL REDACTED [--every 1.0] [--labels bench/labels]
           [--preview bench/reference/preview --n-preview 12] [--thr 35] [--min-area 0.0004]
           [--times 60,600,1500]   # preview only: just dump triptychs at these seconds

Alignment: the copy may be re-encoded at a different frame rate (29.97 -> 30 fps), so frames are
matched by timestamp, not index, and a small constant time offset is estimated first by
searching +-10 frames for the minimum whole-frame difference at a few probe times.

Everything stays local (bench/reference and bench/labels are git-ignored; CJIS).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CLASS_NAME = "redacted"


def _open(path: str) -> tuple[cv2.VideoCapture, float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return cap, fps, n


def _read_at(cap: cv2.VideoCapture, idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx))
    ok, f = cap.read()
    return f if ok else None


def match_tone(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Histogram-match grayscale `src` to `ref`. The reference copy was exported with a different
    tone curve (every mid-tone differed by ~13 levels on the Tanyard pair), which would swamp a
    plain difference; the redacted regions are tiny, so matching on the whole frame is safe."""
    hs = np.bincount(src.ravel(), minlength=256).astype(np.float64)
    hr = np.bincount(ref.ravel(), minlength=256).astype(np.float64)
    cs = np.cumsum(hs) / max(1.0, hs.sum())
    cr = np.cumsum(hr) / max(1.0, hr.sum())
    lut = np.searchsorted(cr, cs).clip(0, 255).astype(np.uint8)
    return lut[src]


def local_std(g: np.ndarray, k: int = 7) -> np.ndarray:
    f = g.astype(np.float32)
    m = cv2.blur(f, (k, k))
    m2 = cv2.blur(f * f, (k, k))
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def diff_mask(origs, red: np.ndarray, thr: int) -> np.ndarray:
    """Binary mask of pixels that the redaction changed.

    origs: 1-3 original frames around the timestamp. Matching by time between 29.97 and 30 fps
    streams is off by up to one frame, which makes the edges of every moving object differ; the
    per-pixel minimum over the neighbouring originals removes that. Three signals are combined:
    strong intensity change (black/colour fills), a weak but spatially consistent change over a
    15x15 area (a blur box over a dark region shifts its mean by only 3-6 levels), and texture loss
    (the original had detail there, the copy has none)."""
    if isinstance(origs, np.ndarray):
        origs = [origs]
    gr = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)
    br = cv2.GaussianBlur(gr, (5, 5), 0)
    sr = local_std(gr, 25)
    ds, edge = [], None
    for o in origs:
        if o.shape[:2] != red.shape[:2]:
            o = cv2.resize(o, (red.shape[1], red.shape[0]), interpolation=cv2.INTER_AREA)
        go = match_tone(cv2.cvtColor(o, cv2.COLOR_BGR2GRAY), gr)
        ds.append(cv2.absdiff(cv2.GaussianBlur(go, (5, 5), 0), br).astype(np.float32))
        so = local_std(go, 25)
        e = (so > 2.0) & (sr < 0.35 * so)
        edge = e if edge is None else (edge & e)
    d = np.min(np.stack(ds), axis=0)
    d_area = cv2.blur(d, (15, 15))
    m = ((d > thr) | ((d_area > 3.5) & (sr < 4.0)) | edge).astype(np.uint8) * 255   # weak shift only where the copy is flat
    k7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    k41 = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k7)      # drop speckle
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k41)    # join a box's fragments
    return m


def boxes_from_mask(mask: np.ndarray, min_area_frac: float) -> list[tuple[int, int, int, int]]:
    H, W = mask.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area_frac * W * H and w >= 6 and h >= 6:
            out.append((int(x), int(y), int(w), int(h)))
    return out


def style_of(red: np.ndarray, box: tuple[int, int, int, int]) -> str:
    x, y, w, h = box
    roi = red[y:y + h, x:x + w]
    if roi.size == 0:
        return "?"
    mean = float(roi.mean())
    lap = float(cv2.Laplacian(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    if mean < 20:
        return "black"
    if lap < 15:
        return "blur"
    return "pixelate/other"


def estimate_offset(co, cr, fps_o, fps_r, probes: list[float], thr: int) -> float:
    """Constant time offset (seconds) to add to the original's timestamp to land on the matching
    redacted frame: minimise the mean absolute difference over +-10 redacted frames."""
    best_total = {}
    for t in probes:
        fo = _read_at(co, int(round(t * fps_o)))
        if fo is None:
            continue
        g_o = cv2.cvtColor(cv2.resize(fo, (320, 180)), cv2.COLOR_BGR2GRAY)
        base = int(round(t * fps_r))
        for k in range(-10, 11):
            fr = _read_at(cr, base + k)
            if fr is None:
                continue
            g_r = cv2.cvtColor(cv2.resize(fr, (320, 180)), cv2.COLOR_BGR2GRAY)
            best_total[k] = best_total.get(k, 0.0) + float(cv2.absdiff(g_o, g_r).mean())
    if not best_total:
        return 0.0
    k = min(best_total, key=best_total.get)
    return k / fps_r


def triptych(orig, red, mask, boxes, title: str) -> np.ndarray:
    o = orig.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(o, (x, y), (x + w, y + h), (0, 255, 0), 2)
    m = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    out = np.hstack([o, red, m])
    out = cv2.resize(out, (out.shape[1] // 2, out.shape[0] // 2), interpolation=cv2.INTER_AREA)
    cv2.putText(out, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("original")
    ap.add_argument("redacted")
    ap.add_argument("--every", type=float, default=1.0, help="sample period in seconds")
    ap.add_argument("--labels", default="bench/labels")
    ap.add_argument("--preview", default="bench/reference/preview")
    ap.add_argument("--n-preview", type=int, default=12)
    ap.add_argument("--thr", type=int, default=20)
    ap.add_argument("--min-area", type=float, default=0.0008)
    ap.add_argument("--times", default="", help="preview only: comma-separated seconds")
    ap.add_argument("--max-seconds", type=float, default=0.0)
    a = ap.parse_args()

    co, fps_o, n_o = _open(a.original)
    cr, fps_r, n_r = _open(a.redacted)
    dur = min(n_o / fps_o, n_r / fps_r)
    stem = Path(a.original).stem.strip()
    prev = Path(a.preview)
    prev.mkdir(parents=True, exist_ok=True)
    print(f"original {n_o} frames @{fps_o:.3f}, redacted {n_r} frames @{fps_r:.3f}, {dur:.1f}s")

    probes = [dur * f for f in (0.1, 0.3, 0.5, 0.7, 0.9)]
    t0 = time.time()
    off = estimate_offset(co, cr, fps_o, fps_r, probes, a.thr)
    print(f"time offset original->redacted: {off * 1000:+.0f} ms ({time.time() - t0:.0f}s)")

    if a.times:
        for t in [float(x) for x in a.times.split(",")]:
            io_ = int(round(t * fps_o))
            origs = [f for f in (_read_at(co, io_ - 1), _read_at(co, io_), _read_at(co, io_ + 1)) if f is not None]
            fo = _read_at(co, io_)
            fr = _read_at(cr, int(round((t + off) * fps_r)))
            if fo is None or fr is None:
                continue
            m = diff_mask(origs, fr, a.thr)
            bx = boxes_from_mask(m, a.min_area)
            styles = [style_of(fr, b) for b in bx]
            print(f"t={t:.0f}s: {len(bx)} boxes {bx} styles={styles}")
            cv2.imwrite(str(prev / f"{stem[:30]}_t{int(t):05d}.png"), triptych(fo, fr, m, bx, f"t={t:.0f}s"))
        return

    # ---- full pass: sequential decode of both streams, sample every `every` seconds ----
    out_dir = Path(a.labels) / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    (Path(a.labels) / "classes.txt").write_text(CLASS_NAME + "\n", encoding="utf-8")
    co.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cr.set(cv2.CAP_PROP_POS_FRAMES, 0)
    io = ir = -1
    fo = fr = None
    stats = {"sampled": 0, "with_boxes": 0, "boxes": 0, "styles": {}, "box_h_px": [], "per_second": []}
    k = 0
    t_end = dur if a.max_seconds <= 0 else min(dur, a.max_seconds)
    H = W = None
    t_start = time.time()
    n_prev = 0
    while True:
        t = k * a.every
        if t > t_end:
            break
        want_o = int(round(t * fps_o))
        want_r = int(round((t + off) * fps_r))
        neigh = {}
        while io < want_o + 1:
            ok = co.grab()
            if not ok:
                break
            io += 1
            if want_o - 1 <= io <= want_o + 1:
                ok, f_ = co.retrieve()
                if ok:
                    neigh[io] = f_
        fo = neigh.get(want_o)
        while ir < want_r:
            ok = cr.grab()
            if not ok:
                fr = None
                break
            ir += 1
            if ir == want_r:
                ok, fr = cr.retrieve()
        if fo is None or fr is None:
            break
        if H is None:
            H, W = fo.shape[:2]
        m = diff_mask([neigh[i] for i in sorted(neigh)], fr, a.thr)
        bx = boxes_from_mask(m, a.min_area)
        stats["sampled"] += 1
        lines = []
        for (x, y, w, h) in bx:
            lines.append(f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
            s = style_of(fr, (x, y, w, h))
            stats["styles"][s] = stats["styles"].get(s, 0) + 1
            stats["box_h_px"].append(int(h))
        (out_dir / f"{stem}_f{want_o:06d}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        stats["boxes"] += len(bx)
        stats["with_boxes"] += bool(bx)
        stats["per_second"].append(len(bx))
        if bx and n_prev < a.n_preview and (stats["with_boxes"] % max(1, 60 // max(1, int(a.every)) ) == 1):
            cv2.imwrite(str(prev / f"{stem[:30]}_f{want_o:06d}.png"), triptych(fo, fr, m, bx, f"t={t:.0f}s f={want_o}"))
            n_prev += 1
        k += 1
        if k % 300 == 0:
            el = time.time() - t_start
            print(f"  {t:7.0f}s / {t_end:.0f}s  boxes so far {stats['boxes']}  ({el:.0f}s, {t / el:.1f}x realtime)", flush=True)
    hs = np.array(stats["box_h_px"]) if stats["box_h_px"] else np.array([0])
    summary = {
        "original": a.original, "redacted": a.redacted, "every_s": a.every, "offset_s": off, "thr": a.thr,
        "min_area": a.min_area, "sampled_frames": stats["sampled"], "frames_with_boxes": stats["with_boxes"],
        "boxes": stats["boxes"], "styles": stats["styles"],
        "box_height_px": {"min": int(hs.min()), "p10": int(np.percentile(hs, 10)), "median": int(np.median(hs)),
                          "p90": int(np.percentile(hs, 90)), "max": int(hs.max())},
        "labels_dir": str(out_dir), "elapsed_s": round(time.time() - t_start, 1),
    }
    (Path(a.labels) / f"{stem}_reference_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("original", "redacted")}, indent=1))


if __name__ == "__main__":
    main()
