"""VideoRedact benchmark: speed per stage and accuracy against ground truth.

    python bench/run_bench.py --clips bench/clips --labels bench/labels --out bench/results/baseline.json
    python bench/run_bench.py --clips bench/clips --labels bench/labels --out bench/results/x.json --labels-classes face,screen

Speed (per clip, single process so stages are attributable; a parallel wall-clock
is measured separately):
  decode  - cv2 frame reads (every frame of the clip)
  detect  - detector inference on stride frames (YuNet + YOLOX)
  track   - association + carry-tracking between detections (auto-detect)
  redact  - redact_frame() on every frame (export path)
  encode  - FFmpeg pipe write + libx264 (export total minus decode/redact)
  export_total, autodetect_total, autodetect_parallel_total

Accuracy (only on frames that have a label file; see bench/README.md):
  Ground truth: YOLO txt per frame (`<cls> cx cy w h`, normalized) in
  bench/labels/<clip_stem>/<clip_stem>_f000123.txt with classes in
  bench/labels/classes.txt, OR a COCO json per clip (bench/labels/<clip_stem>.json).
  Prediction: the redaction boxes the pipeline would apply at that frame
  (auto-detect tracks, interpolated, with each track's padding).
  Metrics per class and overall:
    coverage of a GT box  = fraction of its area inside the union of redaction boxes (grid-sampled)
    recall@90             = GT boxes with coverage >= 0.90 / GT boxes
    precision             = predicted boxes that overlap a GT box of ANY class by >= 50 % of the
                            predicted area / predicted boxes   (over-redaction measure)
    leak_rate             = labelled frames where ANY GT box has coverage < 0.90 / labelled frames
    leak_rate_<class>     = same, restricted to that class
Lower leak rate is the headline number: it is the fraction of sampled frames in which a
viewer could see something that should have been hidden.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from videoredact import __version__  # noqa: E402
from videoredact.core import video_redact  # noqa: E402
from videoredact.core.export import ExportOptions, export_project  # noqa: E402
from videoredact.core.media import FrameReader, probe  # noqa: E402
from videoredact.core.model import BBox, Project, VideoTrack  # noqa: E402
from videoredact.vision import auto_detect  # noqa: E402
from videoredact.vision.auto_detect import AutoDetectOptions, run_auto_detect  # noqa: E402
from videoredact.vision.detector import CombinedDetector  # noqa: E402

DEFAULT_LABELS = ["face", "screen", "document", "phone"]   # what auto-detect is asked to find
GT_CLASSES_DEFAULT = ["face", "screen", "phone_screen", "document", "plate", "id_card"]
COVER_THR = 0.90
GRID = 8


# ----------------------------------------------------------------------------- timing helpers
class StageTimer:
    def __init__(self):
        self.t = defaultdict(float)
        self.n = defaultdict(int)

    def add(self, stage: str, dt: float):
        self.t[stage] += dt
        self.n[stage] += 1


class TimedDetector:
    """Wraps the detector pipeline (vision/detectors) to accumulate inference time."""

    def __init__(self, inner, timer: StageTimer):
        self.inner, self.timer = inner, timer
        self.labels = inner.labels

    def detect_one(self, frame):
        t0 = time.perf_counter()
        out = self.inner.detect_one(frame) if hasattr(self.inner, "detect_one") else self.inner.detect(frame)
        self.timer.add("detect", time.perf_counter() - t0)
        return out

    detect = detect_one


def _patch_timing(timer: StageTimer):
    """Monkeypatch decode (FrameReader.read), carry-tracking (_Carrier) and redact_frame."""
    orig_read = FrameReader.read
    orig_carry_init = auto_detect._Carrier.__init__
    orig_carry_update = auto_detect._Carrier.update
    orig_redact = video_redact.redact_frame

    def read(self):
        t0 = time.perf_counter()
        f = orig_read(self)
        timer.add("decode", time.perf_counter() - t0)
        return f

    def carry_init(self, *a, **k):
        t0 = time.perf_counter()
        orig_carry_init(self, *a, **k)
        timer.add("track", time.perf_counter() - t0)

    def carry_update(self, work):
        t0 = time.perf_counter()
        r = orig_carry_update(self, work)
        timer.add("track", time.perf_counter() - t0)
        return r

    def redact(frame, tracks, idx, style, in_place=True):
        t0 = time.perf_counter()
        r = orig_redact(frame, tracks, idx, style, in_place)
        timer.add("redact", time.perf_counter() - t0)
        return r

    FrameReader.read = read
    auto_detect._Carrier.__init__ = carry_init
    auto_detect._Carrier.update = carry_update
    video_redact.redact_frame = redact

    def restore():
        FrameReader.read = orig_read
        auto_detect._Carrier.__init__ = orig_carry_init
        auto_detect._Carrier.update = orig_carry_update
        video_redact.redact_frame = orig_redact
    return restore


# ----------------------------------------------------------------------------- ground truth
@dataclass
class GT:
    boxes: dict[int, list[tuple[str, BBox]]] = field(default_factory=dict)   # frame -> [(cls, box)]
    labelled_frames: set[int] = field(default_factory=set)


def load_gt(labels_dir: Path, clip_stem: str, classes: list[str]) -> GT:
    gt = GT()
    cdir = labels_dir / clip_stem
    if cdir.is_dir():
        for p in sorted(cdir.glob("*.txt")):
            try:
                frame = int(p.stem.rsplit("_f", 1)[1])
            except (IndexError, ValueError):
                continue
            gt.labelled_frames.add(frame)
            items = []
            for line in p.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                ci = int(float(parts[0]))
                cls = classes[ci] if ci < len(classes) else str(ci)
                cx, cy, w, h = map(float, parts[1:5])
                items.append((cls, BBox(cx - w / 2, cy - h / 2, w, h).clamp()))
            gt.boxes[frame] = items
    cj = labels_dir / f"{clip_stem}.json"
    if cj.exists():
        data = json.loads(cj.read_text(encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in data.get("categories", [])}
        imgs = {}
        for im in data.get("images", []):
            try:
                frame = int(Path(im["file_name"]).stem.rsplit("_f", 1)[1])
            except (IndexError, ValueError):
                continue
            imgs[im["id"]] = (frame, im["width"], im["height"])
            gt.labelled_frames.add(frame)
            gt.boxes.setdefault(frame, [])
        for an in data.get("annotations", []):
            if an["image_id"] not in imgs:
                continue
            frame, W, H = imgs[an["image_id"]]
            x, y, w, h = an["bbox"]
            gt.boxes[frame].append((cats.get(an["category_id"], str(an["category_id"])),
                                    BBox(x / W, y / H, w / W, h / H).clamp()))
    return gt


def coverage(box: BBox, reds: list[BBox]) -> float:
    if not reds:
        return 0.0
    hit = 0
    for i in range(GRID):
        for j in range(GRID):
            px = box.x + (i + 0.5) / GRID * box.w
            py = box.y + (j + 0.5) / GRID * box.h
            if any(r.x <= px <= r.x + r.w and r.y <= py <= r.y + r.h for r in reds):
                hit += 1
    return hit / (GRID * GRID)


def evaluate(tracks: list[VideoTrack], gt: GT, W: int, H: int) -> dict:
    per_cls = defaultdict(lambda: {"gt": 0, "covered": 0, "pred": 0, "pred_hit": 0, "leak_frames": 0})
    frames = 0
    leak_frames = 0
    for f in sorted(gt.labelled_frames):
        frames += 1
        reds, preds = [], []
        for t, bb in video_redact.boxes_at(tracks, f):
            x0, y0, x1, y1 = bb.to_pixels(W, H, t.pad)
            r = BBox.from_pixels(x0, y0, x1, y1, W, H)
            reds.append(r)
            preds.append((t.label, r))
        items = gt.boxes.get(f, [])
        frame_leak = False
        leaked_classes = set()
        for cls, box in items:
            per_cls[cls]["gt"] += 1
            if coverage(box, reds) >= COVER_THR:
                per_cls[cls]["covered"] += 1
            else:
                frame_leak = True
                leaked_classes.add(cls)
        for cls in leaked_classes:
            per_cls[cls]["leak_frames"] += 1
        if frame_leak:
            leak_frames += 1
        gts = [b for _, b in items]
        for label, r in preds:
            per_cls[label]["pred"] += 1
            if gts and coverage(r, gts) >= 0.5:   # >= 50 % of the predicted box lies on some GT box
                per_cls[label]["pred_hit"] += 1
    out = {"labelled_frames": frames, "leak_rate": (leak_frames / frames) if frames else None, "classes": {}}
    tot_gt = tot_cov = tot_pred = tot_hit = 0
    for cls, d in sorted(per_cls.items()):
        out["classes"][cls] = {
            "gt_boxes": d["gt"], "recall@90": (d["covered"] / d["gt"]) if d["gt"] else None,
            "pred_boxes": d["pred"], "precision": (d["pred_hit"] / d["pred"]) if d["pred"] else None,
            "leak_rate": (d["leak_frames"] / frames) if frames else None,
        }
        tot_gt += d["gt"]; tot_cov += d["covered"]; tot_pred += d["pred"]; tot_hit += d["pred_hit"]
    out["recall@90"] = (tot_cov / tot_gt) if tot_gt else None
    out["precision"] = (tot_hit / tot_pred) if tot_pred else None
    return out


# ----------------------------------------------------------------------------- per clip
TRACKS_DIR = Path("bench/tracks")   # detected tracks per clip (git-ignored): re-score without re-detecting


def save_tracks(clip: Path, tracks: list[VideoTrack]) -> Path:
    TRACKS_DIR.mkdir(parents=True, exist_ok=True)
    p = TRACKS_DIR / f"{clip.stem}.json"
    p.write_text(json.dumps([t.to_dict() for t in tracks], default=float), encoding="utf-8")
    return p


def load_tracks(path: Path) -> list[VideoTrack]:
    return [VideoTrack.from_dict(d) for d in json.loads(path.read_text(encoding="utf-8"))]


def _parse_opts(items: list[str]) -> dict:
    """face_pad=0.4 lead_in=30 head_fallback=false -> typed dict for AutoDetectOptions."""
    out = {}
    for it in items:
        k, _, v = it.partition("=")
        vl = v.strip().lower()
        if vl in ("true", "false"):
            out[k.strip()] = (vl == "true")
        else:
            try:
                out[k.strip()] = int(v) if v.strip().lstrip("-").isdigit() else float(v)
            except ValueError:
                out[k.strip()] = v.strip()
    return out


def bench_clip(clip: Path, labels_dir: Path | None, classes: list[str], detect_labels: list[str],
               stride: int, do_export: bool, parallel: bool, tracks_from: Path | None = None,
               detect_kwargs: dict | None = None) -> dict:
    info = probe(str(clip))
    W, H, n = info.width, info.height, info.frame_count
    res = {"clip": clip.name, "width": W, "height": H, "fps": info.fps, "frames": n, "duration_s": info.duration}
    timer = StageTimer()
    restore = _patch_timing(timer)
    try:
        # ---- auto-detect (single process, instrumented) ----
        from videoredact.vision.auto_detect import build_detector
        opts = AutoDetectOptions(labels=detect_labels, stride=stride, workers=1, **(detect_kwargs or {}))
        if tracks_from is not None:
            tracks = load_tracks(tracks_from / f"{clip.stem}.json")
            res["tracks_from"] = str(tracks_from / f"{clip.stem}.json")
            res["autodetect_total_s"] = 0.0
            parallel = False
        else:
            det = TimedDetector(build_detector(opts), timer)
            t0 = time.perf_counter()
            with FrameReader(clip) as r:
                tracks = run_auto_detect(r, opts, detector=det)  # detector given -> single process
            res["autodetect_total_s"] = time.perf_counter() - t0
            res["tracks_file"] = str(save_tracks(clip, tracks))
        res["tracks"] = len(tracks)
        res["track_frames"] = sum(t.total_frames() for t in tracks)
        dec_detect = timer.t["decode"]
        res["stage_s"] = {"decode_detect_pass": round(dec_detect, 3), "detect": round(timer.t["detect"], 3),
                          "track": round(timer.t["track"], 3), "detect_calls": timer.n["detect"]}
        res["stage_s"]["assoc_other"] = round(max(0.0, res["autodetect_total_s"] - dec_detect - timer.t["detect"] - timer.t["track"]), 3)
        # ---- export (instrumented) ----
        if do_export:
            proj = Project(media=info, author="bench")
            for t in tracks:
                proj.video_tracks.append(t)
            out = Path(tempfile.gettempdir()) / f"bench_{clip.stem}.mp4"
            timer.t["decode"] = 0.0
            t0 = time.perf_counter()
            export_project(proj, ExportOptions(str(out), crf=23, preset="veryfast", write_report=False, write_project=False))
            tot = time.perf_counter() - t0
            res["export_total_s"] = tot
            res["stage_s"].update({"decode_export_pass": round(timer.t["decode"], 3), "redact": round(timer.t["redact"], 3),
                                   "encode_and_pipe": round(tot - timer.t["decode"] - timer.t["redact"], 3)})
            try:
                out.unlink()
            except OSError:
                pass
    finally:
        restore()
    res["fps"] = {"autodetect": round(n / res["autodetect_total_s"], 1) if res["autodetect_total_s"] else None}
    if do_export:
        res["fps"]["export"] = round(n / res["export_total_s"], 1)
    # ---- parallel wall clock (what the user experiences) ----
    if parallel:
        t0 = time.perf_counter()
        with FrameReader(clip) as r:
            run_auto_detect(r, AutoDetectOptions(labels=detect_labels, stride=stride, workers=0))
        res["autodetect_parallel_total_s"] = time.perf_counter() - t0
        res["fps"]["autodetect_parallel"] = round(n / res["autodetect_parallel_total_s"], 1)
    # ---- accuracy ----
    if labels_dir is not None:
        gt = load_gt(labels_dir, clip.stem, classes)
        if gt.labelled_frames:
            res["accuracy"] = evaluate(tracks, gt, W, H)
        else:
            res["accuracy"] = {"labelled_frames": 0, "note": "no labels for this clip"}
    return res


def aggregate(clips: list[dict]) -> dict:
    agg = {"clips": len(clips), "frames": sum(c["frames"] for c in clips)}
    tot_auto = sum(c["autodetect_total_s"] for c in clips)
    agg["autodetect_fps"] = round(agg["frames"] / tot_auto, 1) if tot_auto else None
    if all("export_total_s" in c for c in clips):
        agg["export_fps"] = round(agg["frames"] / sum(c["export_total_s"] for c in clips), 1)
    if all("autodetect_parallel_total_s" in c for c in clips):
        agg["autodetect_parallel_fps"] = round(agg["frames"] / sum(c["autodetect_parallel_total_s"] for c in clips), 1)
    stages = defaultdict(float)
    for c in clips:
        for k, v in c["stage_s"].items():
            if k != "detect_calls":
                stages[k] += v
    agg["stage_share"] = {k: round(v, 2) for k, v in stages.items()}
    labelled = [c for c in clips if c.get("accuracy", {}).get("labelled_frames")]
    if labelled:
        frames = sum(c["accuracy"]["labelled_frames"] for c in labelled)
        leak = sum(c["accuracy"]["leak_rate"] * c["accuracy"]["labelled_frames"] for c in labelled)
        per = defaultdict(lambda: {"gt": 0, "cov": 0, "pred": 0, "hit": 0, "leak": 0})
        for c in labelled:
            n = c["accuracy"]["labelled_frames"]
            for cls, d in c["accuracy"]["classes"].items():
                per[cls]["gt"] += d["gt_boxes"]
                per[cls]["cov"] += round((d["recall@90"] or 0) * d["gt_boxes"])
                per[cls]["pred"] += d["pred_boxes"]
                per[cls]["hit"] += round((d["precision"] or 0) * d["pred_boxes"])
                per[cls]["leak"] += round((d["leak_rate"] or 0) * n)
        agg["accuracy"] = {"labelled_frames": frames, "leak_rate": round(leak / frames, 4),
                           "classes": {cls: {"gt_boxes": d["gt"], "recall@90": round(d["cov"] / d["gt"], 4) if d["gt"] else None,
                                             "pred_boxes": d["pred"], "precision": round(d["hit"] / d["pred"], 4) if d["pred"] else None,
                                             "leak_rate": round(d["leak"] / frames, 4)} for cls, d in sorted(per.items())}}
    else:
        agg["accuracy"] = {"labelled_frames": 0, "note": "no ground truth yet - run the labeling workflow in bench/README.md"}
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="bench/clips")
    ap.add_argument("--labels", default="bench/labels")
    ap.add_argument("--out", default="bench/results/run.json")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--detect-labels", default=",".join(DEFAULT_LABELS))
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--no-parallel", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips")
    ap.add_argument("--match", default="", help="only clips whose file name contains this text")
    ap.add_argument("--tracks-from", default="", help="score tracks saved by an earlier run (dir of <stem>.json) instead of detecting")
    ap.add_argument("--detect-opt", action="append", default=[], help="AutoDetectOptions override, e.g. face_pad=0.4 (repeatable)")
    a = ap.parse_args()
    clips = sorted(p for p in Path(a.clips).glob("*.mp4") if a.match.lower() in p.name.lower())
    if a.limit:
        clips = clips[: a.limit]
    labels_dir = Path(a.labels) if Path(a.labels).exists() else None
    classes = GT_CLASSES_DEFAULT
    cf = Path(a.labels) / "classes.txt"
    if cf.exists():
        classes = [l.strip() for l in cf.read_text(encoding="utf-8").splitlines() if l.strip()]
    from videoredact.vision.accel import detect_hardware
    hw = detect_hardware()
    results = {"version": __version__, "date": time.strftime("%Y-%m-%d %H:%M"), "hardware": hw.summary(),
               "python": platform.python_version(), "settings": {"stride": a.stride, "detect_labels": a.detect_labels.split(","),
               "cover_threshold": COVER_THR, "gt_classes": classes}, "clips": []}
    for clip in clips:
        print(f"== {clip.name}", flush=True)
        r = bench_clip(clip, labels_dir, classes, a.detect_labels.split(","), a.stride, not a.no_export, not a.no_parallel,
                       tracks_from=Path(a.tracks_from) if a.tracks_from else None, detect_kwargs=_parse_opts(a.detect_opt))
        acc = r.get("accuracy", {})
        print(f"   autodetect {r['fps']['autodetect']} fps (parallel {r['fps'].get('autodetect_parallel', '-')}), "
              f"export {r['fps'].get('export', '-')} fps, tracks={r['tracks']}, "
              f"leak_rate={acc.get('leak_rate')} on {acc.get('labelled_frames', 0)} labelled frames", flush=True)
        results["clips"].append(r)
    results["aggregate"] = aggregate(results["clips"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("\nAGGREGATE:", json.dumps(results["aggregate"], indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
