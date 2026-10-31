"""Score a candidate redaction against a hand-made one (the ground truth Wyatt produces in the app).

Both sides are VideoRedact project files (.vrproj) for the same media. The reference is the human's
project: every enabled video track in it marks something that must be hidden. The candidate is
either another project file, or produced here from the same media:

  --auto      run Auto-detect with the current settings (faces by default) and score its tracks
  --retrack   draw ONE box per reference track (its box at the middle of the track) and run the
              single-object tracker the way the app does; scores "one box covers the object's
              whole visible life", the Phase 1 goal

Per sampled frame (every --step frames) and per reference box: covered when the union of the
candidate's padded boxes hides at least --cover (default 0.85) of the reference box's area
(rasterised on a 480 px grid). A reference box that is not covered is a leak.

  recall      share of reference box-frames covered (by label, and overall)
  leak runs   consecutive frames with a leak, longest first, so they can be reviewed in the app
  over        candidate box-frames that overlap no reference box at all (over-redaction)
  extra area  share of the frame area the candidate hides that the reference does not

Usage:
  python bench/score_project.py --ref my_hand_redaction.vrproj --cand auto_or_tracked.vrproj
  python bench/score_project.py --ref my_hand_redaction.vrproj --auto
  python bench/score_project.py --ref my_hand_redaction.vrproj --retrack [--engine anchored|classic]
  add --sheet leaks.jpg to write crops of the worst leak frames, --out result.json to save numbers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from videoredact.core.model import BBox, Project, VideoTrack  # noqa: E402

GRID_W = 480


def _mask(boxes, W, H, pad):
    gh = max(1, int(round(GRID_W * H / W)))
    m = np.zeros((gh, GRID_W), bool)
    for b in boxes:
        x0, y0, x1, y1 = b.to_pixels(GRID_W, gh, pad=pad)
        m[y0:y1, x0:x1] = True
    return m


def active(tracks, frame):
    out = []
    for t in tracks:
        if not t.enabled:
            continue
        b = t.bbox_at(frame)
        if b is not None:
            out.append((t, b))
    return out


def score(ref: Project, cand_tracks, step: int, cover: float, W: int, H: int):
    ref_tracks = [t for t in ref.video_tracks if t.enabled]
    last = max([t.last_frame() for t in ref_tracks] + [t.last_frame() for t in cand_tracks if t.enabled] + [0])
    gh = max(1, int(round(GRID_W * H / W)))
    by_label = defaultdict(lambda: dict(boxes=0, covered=0))
    leaks = []            # (frame, label, track id, covered fraction)
    over_frames = 0
    extra_area = 0.0
    frames = 0
    for f in range(0, last + 1, step):
        r = active(ref_tracks, f)
        c = active(cand_tracks, f)
        if not r and not c:
            continue
        frames += 1
        cmask = _mask([b for _, b in c], W, H, pad=max((t.pad for t, _ in c), default=0.0)) if c else np.zeros((gh, GRID_W), bool)
        rmask = _mask([b for _, b in r], W, H, pad=0.0) if r else np.zeros((gh, GRID_W), bool)
        for t, b in r:
            x0, y0, x1, y1 = b.to_pixels(GRID_W, gh, pad=0.0)
            area = max(1, (x1 - x0) * (y1 - y0))
            frac = float(cmask[y0:y1, x0:x1].sum()) / area
            by_label[t.label]["boxes"] += 1
            if frac >= cover:
                by_label[t.label]["covered"] += 1
            else:
                leaks.append((f, t.label, t.id, round(frac, 2)))
        if c:
            if not any(rmask[b.to_pixels(GRID_W, gh)[1]:b.to_pixels(GRID_W, gh)[3], b.to_pixels(GRID_W, gh)[0]:b.to_pixels(GRID_W, gh)[2]].any() for _, b in c):
                over_frames += 1
            extra_area += float((cmask & ~rmask).sum()) / cmask.size
    # leak runs: consecutive sampled frames of the same track
    runs = []
    for tid in {l[2] for l in leaks}:
        fs = sorted(l[0] for l in leaks if l[2] == tid)
        label = next(l[1] for l in leaks if l[2] == tid)
        start = prev = fs[0]
        for f in fs[1:] + [None]:
            if f is None or f - prev > step:
                runs.append(dict(label=label, track=tid, start=start, end=prev, frames=prev - start + step))
                if f is not None:
                    start = f
            if f is not None:
                prev = f
    runs.sort(key=lambda r: -r["frames"])
    total_boxes = sum(v["boxes"] for v in by_label.values())
    total_cov = sum(v["covered"] for v in by_label.values())
    return dict(frames_sampled=frames, step=step, cover=cover,
                recall=round(total_cov / total_boxes, 4) if total_boxes else None,
                by_label={k: dict(v, recall=round(v["covered"] / v["boxes"], 4)) for k, v in by_label.items()},
                leak_boxes=total_boxes - total_cov, leak_runs=runs[:50],
                over_frames=over_frames, extra_area_mean=round(extra_area / max(1, frames), 4))


def retrack(ref: Project, media: str, engine: str, progress=print):
    """One drawn box per reference track (its box at the middle), tracked both ways like the app."""
    from videoredact.core.media import FrameReader
    from videoredact.paths import DEFAULT_SETTINGS as S
    from videoredact.vision.detector import CombinedDetector
    from videoredact.vision.tracker import ObjectTracker, TrackOptions
    out = []
    with FrameReader(media) as reader:
        for t in ref.video_tracks:
            if not t.enabled or not t.spans:
                continue
            # the drawn box: the middle of the track's longest span (a track's overall middle can
            # fall in a gap where the object is not even on screen)
            span = max(t.spans, key=lambda s: s.end_frame - s.start_frame)
            mid = (span.start_frame + span.end_frame) // 2
            b = t.bbox_at(mid)
            if b is None:
                continue
            det = None
            if t.label in ("face", "screen", "document", "phone", "person"):
                det = CombinedDetector([t.label], S["face_conf"], S["detect_conf"], S.get("detector_model", "yolox_tiny"), 0,
                                       face_model=S.get("face_detector", "yunet"))
            t0 = time.time()
            tr = ObjectTracker(reader, det, TrackOptions(engine=engine)).track(mid, b, label=t.label)
            tr.pad = t.pad
            tr.note = f"retracked from frame {mid} of reference track {t.id}"
            out.append(tr)
            progress(f"  {t.label} ref {t.first_frame()}-{t.last_frame()} -> tracked {tr.total_frames()} frames in "
                     f"{len(tr.spans)} span(s), {time.time() - t0:.1f}s")
    return out


def auto(ref: Project, media: str, progress=print):
    from videoredact.core.media import FrameReader
    from videoredact.paths import DEFAULT_SETTINGS as S
    from videoredact.vision.auto_detect import AutoDetectOptions, run_auto_detect
    labels = sorted({t.label for t in ref.video_tracks if t.label in ("face", "screen", "document", "phone", "person")}) or ["face"]
    with FrameReader(media) as reader:
        opts = AutoDetectOptions(labels=labels)
        opts.apply_profile(bool(S.get("detect_precise", True)))
        tracks = run_auto_detect(reader, opts, progress=lambda p, m: None)
    progress(f"  auto-detect ({', '.join(labels)}): {len(tracks)} tracks")
    return tracks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="hand-made .vrproj (ground truth)")
    ap.add_argument("--cand", help="candidate .vrproj")
    ap.add_argument("--auto", action="store_true", help="candidate = Auto-detect on the reference's media")
    ap.add_argument("--retrack", action="store_true", help="candidate = one tracked box per reference track")
    ap.add_argument("--engine", default="anchored")
    ap.add_argument("--media", help="override the media path stored in the reference project")
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--cover", type=float, default=0.85)
    ap.add_argument("--out", default="")
    ap.add_argument("--save-cand", default="", help="save the generated candidate as a .vrproj for review in the app")
    ap.add_argument("--sheet", default="", help="contact sheet of the worst leak frames (needs the media)")
    a = ap.parse_args()
    ref = Project.load(a.ref)
    media = a.media or ref.media.path
    W, H = ref.media.width, ref.media.height
    if a.cand:
        cand = Project.load(a.cand).video_tracks
        src = a.cand
    elif a.retrack:
        cand = retrack(ref, media, a.engine)
        src = f"retrack ({a.engine})"
    elif a.auto:
        cand = auto(ref, media)
        src = "auto-detect"
    else:
        raise SystemExit("give --cand, --auto or --retrack")
    if a.save_cand:
        p = Project.load(a.ref)
        p.video_tracks = cand
        p.save(a.save_cand)
        print("candidate saved to", a.save_cand)
    res = score(ref, cand, a.step, a.cover, W, H)
    res["reference"], res["candidate"] = a.ref, src
    print(f"reference {os.path.basename(a.ref)}: {sum(1 for t in ref.video_tracks if t.enabled)} tracks; candidate {src}: {len(cand)} tracks")
    print(f"recall {res['recall']:.1%} of {sum(v['boxes'] for v in res['by_label'].values())} reference box-frames "
          f"(sampled every {a.step} frames, covered means >= {a.cover:.0%} of the box hidden)")
    for k, v in sorted(res["by_label"].items()):
        print(f"   {k:14} {v['recall']:.1%}  ({v['covered']}/{v['boxes']})")
    print(f"over-redaction: {res['over_frames']} sampled frames with boxes on nothing marked; "
          f"extra hidden area {res['extra_area_mean']:.1%} of the frame on average")
    if res["leak_runs"]:
        print("longest leaks (frames):")
        for r in res["leak_runs"][:12]:
            print(f"   {r['label']:14} {r['start']:6}-{r['end']:6}  {r['frames']} frames  track {r['track']}")
    if a.sheet and res["leak_runs"]:
        import cv2
        from videoredact.core.media import FrameReader
        tiles = []
        with FrameReader(media) as reader:
            for r in res["leak_runs"][:40]:
                f = (r["start"] + r["end"]) // 2
                t = next((t for t in ref.video_tracks if t.id == r["track"]), None)
                img = reader.read_at(f)
                b = t.bbox_at(f) if t else None
                if img is None or b is None:
                    continue
                x0, y0, x1, y1 = b.to_pixels(W, H, pad=0.8)
                crop = cv2.resize(img[y0:y1, x0:x1], (200, 200)) if (y1 > y0 and x1 > x0) else None
                if crop is not None:
                    cv2.putText(crop, f"{r['label']} f{f}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                    tiles.append(crop)
        if tiles:
            cols = 8
            tiles += [np.zeros((200, 200, 3), np.uint8)] * ((cols - len(tiles) % cols) % cols)
            cv2.imwrite(a.sheet, np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))
            print("wrote", a.sheet)
    if a.out:
        with open(a.out, "w", newline="\n") as fh:
            json.dump(res, fh, indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
