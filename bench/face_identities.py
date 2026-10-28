"""How many *people* are behind the face tracks? Prototype of per-person grouping.

Samples a few frames from every face track saved by bench/run_bench.py, re-detects the face
with YuNet to get landmarks, embeds it with SFace (OpenCV Zoo, Apache-2.0) and clusters the
embeddings (scipy average-linkage, cosine; embeddings cached in --out for threshold sweeps). Prints cluster sizes and writes a contact sheet per cluster
so a person can check that one cluster = one person. This is the measurement behind the
"redact everyone except..." / "this person everywhere" feature (ROADMAP item 4).

    python bench/face_identities.py bench/reference/tanyard_original.mp4 [--tracks bench/tracks/X.json]
           [--samples 3] [--thr 0.40] [--out bench/reference/identities]

SFace model: downloaded once into the user models folder (dev machine; not shipped yet).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from videoredact.paths import models_dir, model_search_dirs  # noqa: E402
from run_bench import load_tracks  # noqa: E402

SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"


def _find(name: str) -> Path | None:
    for d in model_search_dirs():
        p = d / name
        if p.exists():
            return p
    return None


def embed_tracks(tracks, cap, W, H, det, rec, a):
    t0 = time.time()
    feats, owners, crops = [], [], []
    skipped = 0
    for ti, t in enumerate(tracks):
        f0, f1 = t.first_frame(), t.last_frame()
        frames = sorted({int(f0 + (f1 - f0) * k / max(1, a.samples - 1)) for k in range(a.samples)})
        for f in frames:
            bb = t.bbox_at(f)
            if bb is None:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                continue
            x0, y0, x1, y1 = bb.to_pixels(W, H, 0.6)   # generous crop so YuNet sees the whole head
            if x1 - x0 < a.min_face or y1 - y0 < a.min_face:
                skipped += 1
                continue
            crop = img[y0:y1, x0:x1]
            det.setInputSize((crop.shape[1], crop.shape[0]))
            _, faces = det.detect(crop)
            if faces is None or len(faces) == 0:
                skipped += 1
                continue
            face = max(faces, key=lambda r: r[2] * r[3])
            aligned = rec.alignCrop(crop, face)
            feat = rec.feature(aligned)
            feats.append(feat.flatten() / (np.linalg.norm(feat) + 1e-9))
            owners.append(ti)
            crops.append(cv2.resize(aligned, (56, 56)))
        if (ti + 1) % 200 == 0:
            print(f"  {ti + 1}/{len(tracks)} tracks, {len(feats)} embeddings ({time.time() - t0:.0f}s)", flush=True)
    print(f"{len(tracks)} face tracks -> {len(feats)} embeddings ({skipped} samples unusable) in {time.time() - t0:.0f}s")
    return (np.stack(feats) if feats else np.zeros((0, 128), np.float32)), owners, crops


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--tracks", default="")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--thr", type=float, default=0.70,
                    help="cosine distance for 'same person'. SFace's published match point is 0.64; on a 45-min body-cam "
                         "clip 0.70 gave three pure clusters for the three real people, 0.40 split them into hundreds")
    ap.add_argument("--out", default="bench/reference/identities")
    ap.add_argument("--min-face", type=int, default=24)
    ap.add_argument("--recompute", action="store_true", help="ignore the cached embeddings in --out")
    a = ap.parse_args()
    clip = Path(a.video)
    tracks = [t for t in load_tracks(Path(a.tracks) if a.tracks else Path("bench/tracks") / f"{clip.stem}.json") if t.label == "face"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    sface = _find("face_recognition_sface_2021dec.onnx")
    if sface is None:
        sface = models_dir() / "face_recognition_sface_2021dec.onnx"
        print("downloading SFace to", sface)
        urllib.request.urlretrieve(SFACE_URL, sface)
    yunet = _find("face_detection_yunet_2023mar.onnx")
    det = cv2.FaceDetectorYN.create(str(yunet), "", (320, 320), 0.5, 0.3, 5000)
    rec = cv2.FaceRecognizerSF.create(str(sface), "")

    cap = cv2.VideoCapture(str(clip))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cache = out / f"{clip.stem}_embeddings.npz"
    if cache.exists() and not a.recompute:
        z = np.load(cache)
        X, owners, crops = z["X"], z["owners"].tolist(), list(z["crops"])
        print(f"loaded {len(X)} cached embeddings from {cache}")
    else:
        X, owners, crops = embed_tracks(tracks, cap, W, H, det, rec, a)
        if len(X) == 0:
            return
        np.savez_compressed(cache, X=X, owners=np.array(owners), crops=np.stack(crops))
    n = len(X)
    # agglomerative clustering, average linkage on cosine distance (scipy; dev tool only)
    from scipy.cluster.hierarchy import fcluster, linkage
    t0 = time.time()
    Z = linkage(X, method="average", metric="cosine")
    labels = fcluster(Z, t=a.thr, criterion="distance")
    clusters = defaultdict(list)
    for i, lab in enumerate(labels):
        clusters[int(lab)].append(i)
    clusters = list(clusters.values())
    print(f"clustered {n} embeddings in {time.time() - t0:.1f}s")
    # one person per track: majority cluster of its samples
    track_cluster = {}
    for ci, members in enumerate(clusters):
        for m in members:
            track_cluster.setdefault(owners[m], []).append(ci)
    person_of = {ti: max(set(cs), key=cs.count) for ti, cs in track_cluster.items()}
    by_person = defaultdict(list)
    for ti, p in person_of.items():
        by_person[p].append(ti)
    sizes = sorted(((len(v), p) for p, v in by_person.items()), reverse=True)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"{len(clusters)} clusters from {n} embeddings; {len(by_person)} people over {len(person_of)} tracks")
    print("largest people (tracks, total minutes on screen):")
    for sz, p in sizes[:12]:
        mins = sum(tracks[ti].total_frames() for ti in by_person[p]) / fps / 60
        print(f"  person {p:4d}: {sz:5d} tracks, {mins:6.1f} min")
    small = sum(1 for sz, _ in sizes if sz == 1)
    print(f"singleton people (1 track): {small}")
    # contact sheets for the biggest people
    for sz, p in sizes[:8]:
        members = [i for i in range(n) if person_of.get(owners[i]) == p][:48]
        tiles = [crops[i] for i in members]
        while len(tiles) % 8:
            tiles.append(np.zeros((56, 56, 3), np.uint8))
        rows = [np.hstack(tiles[r:r + 8]) for r in range(0, len(tiles), 8)]
        cv2.imwrite(str(out / f"person_{p:04d}_{sz}tracks.png"), np.vstack(rows))
    (out / f"{clip.stem}_identities.json").write_text(json.dumps({"tracks": len(tracks), "embeddings": n, "people": len(by_person),
                                                                   "sizes": [sz for sz, _ in sizes], "thr": a.thr}), encoding="utf-8")


if __name__ == "__main__":
    main()
