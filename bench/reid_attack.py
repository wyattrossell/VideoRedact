"""Re-identification attack on a redacted export: can a face recogniser still match the
people in the output to the people in the original?

Detection benchmarks ask "did a box fire?". The question a records officer or a court asks
is "can this person still be identified from the released file?". This harness attacks the
redacted file directly, the way an adversary with the original's geometry would:

  1. ALIGN      source and redacted output are read frame by frame in lockstep (our export keeps
                the frame count, size and rate of the source; a mismatch refuses to grade).
  2. CANDIDATES YuNet on the *source* at a low threshold (0.30) over-finds candidates; only
                those confirmed at the normal threshold (0.60) and at least 32 px across are
                attacked (small or doubtful crops carry no identity to leak or to protect).
  3. TRACKS     confirmed faces are IoU-linked over time into per-person tracks.
  4. CONTROL    original-vs-original similarity within a track. If SFace cannot match an
                UNREDACTED face to another frame of the same person, the clip cannot support a
                re-identification claim and the harness says so instead of printing a number.
  5. ATTACK     for every confirmed face: SFace is aligned on the *source's* landmarks and run on
                the *redacted* pixels (the strongest alignment an attacker could have); is the
                embedding still the same identity (cosine >= the app's own same-person
                threshold, and the stricter 0.55 OpenScrub uses)? Is a face still detectable
                there? Did the pixels change at all? An unchanged face is a pipeline MISS (or a
                face the reviewer chose to leave visible, when a project file says so); a changed
                face that still matches is WEAK redaction. Different bugs, reported separately.
  6. FLAT FLOOR a crop whose original embedding matches a flat patch of its own mean colour at
                >= 0.5 is colour mass, not a face; it is excluded and counted openly.

Protocol after OpenScrub's benchmark/reid.py (Apache-2.0), re-implemented on our own detector,
identity model and track model. Footage never leaves the machine; only the numbers are kept.

    python bench/reid_attack.py --source clip.mp4 --redacted clip_black.mp4 [--project clip_black.vrproj]
                                [--stride 15] [--out bench/results/reid_clip_black.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from videoredact.core.media import FrameReader  # noqa: E402
from videoredact.core.model import BBox, Project  # noqa: E402
from videoredact.core.verify import _coverage  # noqa: E402
from videoredact.core.video_redact import boxes_at  # noqa: E402
from videoredact.vision.detector import CombinedDetector  # noqa: E402
from videoredact.vision.identity import SAME_PERSON, FaceIdentity  # noqa: E402

CANDIDATE_THR = 0.30
CONFIRM_THR = 0.60
MIN_FACE_PX = 32
STRICT_THR = 0.55          # OpenScrub's same-person point for SFace; ours (0.363) is the documented OpenCV one
FLAT_FLOOR = 0.5
CHANGED_MIN_DIFF = 2.0     # mean absolute pixel difference below this = the crop was not touched
CONTROL_MIN = 0.5
LINK_IOU = 0.3


def _cos(a, b) -> float:
    return float(np.dot(a, b))


def _crop(frame, bbox: BBox, pad: float = 0.0):
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = bbox.to_pixels(W, H, pad)
    return frame[y0:y1, x0:x1], (x0, y0, x1, y1)


def attack(source: str, redacted: str, project: Project | None = None, stride: int = 15,
           progress=print) -> dict:
    det = CombinedDetector(["face"], face_conf=CANDIDATE_THR)
    det.set_score_threshold(CANDIDATE_THR)
    identity = FaceIdentity.try_create()
    if identity is None:
        raise RuntimeError("SFace model not installed (run scripts/fetch_models.py)")
    tracks = [t for t in project.video_tracks if t.enabled] if project else None
    rows = []
    unconfirmed = 0
    t0 = time.time()
    with FrameReader(source) as rs, FrameReader(redacted) as rr:
        if (rs.width, rs.height) != (rr.width, rr.height):
            raise RuntimeError(f"size mismatch: source {rs.width}x{rs.height}, redacted {rr.width}x{rr.height}")
        if abs(rs.frame_count - rr.frame_count) > 2:
            raise RuntimeError(f"frame count mismatch: source {rs.frame_count}, redacted {rr.frame_count}; "
                               "the attack must compare like with like (clip exports are not supported)")
        W, H = rs.width, rs.height
        fps = rs.fps or 30.0
        n = min(rs.frame_count, rr.frame_count)
        live: list[dict] = []          # IoU linking state: {"box", "frame", "id"}
        next_id = 0
        for idx in range(0, n, stride):
            fs, fr = rs.read_at(idx), rr.read_at(idx)
            if fs is None or fr is None:
                break
            cands = [d for d in det.detect(fs) if d.label == "face" and d.landmarks is not None]
            # link confirmed faces into tracks (one sample per track per frame)
            new_live = []
            for d in cands:
                if d.score < CONFIRM_THR or d.bbox.w * W < MIN_FACE_PX or d.bbox.h * H < MIN_FACE_PX:
                    unconfirmed += 1
                    continue
                best, best_iou = None, LINK_IOU
                for L in live:
                    if L["frame"] == idx:
                        continue
                    iou = d.bbox.iou(L["box"])
                    if iou > best_iou and L not in new_live:
                        best, best_iou = L, iou
                if best is None:
                    best = {"id": next_id}
                    next_id += 1
                best.update(box=d.bbox, frame=idx)
                new_live.append(best)
                row = {"frame": idx, "time_s": round(idx / fps, 2), "track": best["id"], "score": round(float(d.score), 3),
                       "box": [round(v, 4) for v in d.bbox.to_list()], "px": int(d.bbox.w * W)}
                e_src = identity.embed(fs, d)
                if e_src is None:
                    row["status"] = "no-embedding"
                    rows.append(row)
                    continue
                crop_s, _ = _crop(fs, d.bbox)
                flat = np.empty_like(fs)
                flat[:] = crop_s.reshape(-1, 3).mean(axis=0).astype(np.uint8)
                e_flat = identity.embed(flat, d)
                row["flat_floor"] = round(_cos(e_src, e_flat), 3) if e_flat is not None else None
                crop_r, (x0, y0, x1, y1) = _crop(fr, d.bbox)
                row["changed"] = bool(float(np.mean(cv2.absdiff(crop_s, crop_r))) >= CHANGED_MIN_DIFF)
                if tracks is not None:
                    red = [BBox.from_pixels(*bb.to_pixels(W, H, t.pad), W, H) for t, bb in boxes_at(tracks, idx)]
                    row["intended"] = _coverage(d.bbox, red) >= 0.6
                e_red = identity.embed(fr, d)
                row["sim"] = round(_cos(e_src, e_red), 3) if e_red is not None else None
                # is anything face-like still detectable where the face was?
                region, (rx0, ry0, rx1, ry1) = _crop(fr, d.bbox, 0.5)
                still = False
                if region.size:
                    for q in det.detect(region):
                        qb = BBox((rx0 + q.bbox.x * region.shape[1]) / W, (ry0 + q.bbox.y * region.shape[0]) / H,
                                  q.bbox.w * region.shape[1] / W, q.bbox.h * region.shape[0] / H)
                        if q.label == "face" and q.score >= CONFIRM_THR and qb.iou(d.bbox) >= LINK_IOU:
                            still = True
                            break
                row["still_detected"] = still
                row["_e"] = e_src
                rows.append(row)
            live = new_live + [L for L in live if L not in new_live and idx - L["frame"] <= 2 * stride]
            if idx % (stride * 20) == 0:
                progress(f"  frame {idx}/{n}: {len(rows)} faces attacked so far")
    # ---- control: original vs original inside each track; cross-person floor between tracks ----
    by_track: dict[int, list] = {}
    for r in rows:
        if "_e" in r:
            by_track.setdefault(r["track"], []).append(r)
    within = []
    for rs_ in by_track.values():
        for a, b in zip(rs_, rs_[1:]):
            within.append(_cos(a["_e"], b["_e"]))
    cross = []
    tids = list(by_track)
    for i in range(len(tids)):
        for j in range(i + 1, len(tids)):
            for a in by_track[tids[i]][:3]:
                for b in by_track[tids[j]][:3]:
                    cross.append(_cos(a["_e"], b["_e"]))
    control = float(np.mean(within)) if within else None
    control_ok = control is not None and control >= CONTROL_MIN
    # ---- summarise ----
    attacked = [r for r in rows if r.get("sim") is not None and r.get("flat_floor") is not None]
    degenerate = [r for r in attacked if r["flat_floor"] >= FLAT_FLOOR]
    graded = [r for r in attacked if r["flat_floor"] < FLAT_FLOOR]
    left_by_policy = [r for r in graded if r.get("intended") is False]
    target = [r for r in graded if r.get("intended") is not False]
    missed = [r for r in target if not r["changed"]]
    changed = [r for r in target if r["changed"]]
    reid = [r for r in changed if r["sim"] >= SAME_PERSON]
    reid_strict = [r for r in changed if r["sim"] >= STRICT_THR]
    still = [r for r in changed if r["still_detected"]]
    for r in rows:
        r.pop("_e", None)
    summary = {
        "source": source, "redacted": redacted, "project": None if project is None else True,
        "stride": stride, "frames_sampled": (n + stride - 1) // stride, "elapsed_s": round(time.time() - t0, 1),
        "candidates_unconfirmed": unconfirmed, "faces_confirmed": len(rows), "tracks": len(by_track),
        "control_same_person_similarity": None if control is None else round(control, 3),
        "control_pairs": len(within), "control_valid": control_ok,
        "cross_person_similarity": None if not cross else round(float(np.mean(cross)), 3),
        "degenerate_excluded": len(degenerate), "left_visible_by_policy": len(left_by_policy),
        "faces_attacked": len(target), "unchanged_pixels_miss": len(missed), "redacted_faces": len(changed),
        "reidentified_at_app_threshold": len(reid), "reidentified_at_0.55": len(reid_strict),
        "still_detected_as_face": len(still),
        "mean_similarity_after_redaction": None if not changed else round(float(np.mean([r["sim"] for r in changed])), 3),
        "thresholds": {"app_same_person": SAME_PERSON, "strict": STRICT_THR, "confirm": CONFIRM_THR,
                       "candidate": CANDIDATE_THR, "min_face_px": MIN_FACE_PX, "flat_floor": FLAT_FLOOR},
        "verdict": ("CONTROL INVALID: the recogniser cannot re-identify unredacted faces in this clip, so "
                    "a re-identification rate here would be meaningless" if not control_ok else
                    f"{len(reid)} of {len(changed)} redacted faces re-identified at {SAME_PERSON}; "
                    f"{len(reid_strict)} at {STRICT_THR}; {len(still)} still detected as a face; "
                    f"{len(missed)} face-frames left untouched (pipeline miss)"),
    }
    return {"summary": summary, "faces": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--redacted", required=True)
    ap.add_argument("--project", help=".vrproj of the export: faces its tracks do not cover count as 'left visible by policy'")
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    proj = Project.load(a.project) if a.project else None
    res = attack(a.source, a.redacted, proj, a.stride)
    s = res["summary"]
    print(json.dumps(s, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print("wrote", a.out)


if __name__ == "__main__":
    main()
