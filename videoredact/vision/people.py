"""Same-person matching across face tracks (Phase 3b of the tracking rebuild, 2026-10-31).

A reviewer selects one face track and asks "find this person everywhere": every other face track
is embedded with SFace at a few of its frames and compared with the subject's embeddings. The
result drives bulk actions (disable / delete the matches, or keep only this person redacted).
Everything runs locally; embeddings live in memory for the duration of the job.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from videoredact.core.media import FrameReader
from videoredact.core.model import VideoTrack
from .identity import SAME_PERSON, FaceIdentity

ProgressCB = Callable[[float, str], None]
CancelCB = Callable[[], bool]

SAMPLES = 5           # frames embedded per track, spread over its life
CROP_PAD = 0.6        # the track box is padded so the detector sees the whole head
MIN_FACE_PX = 24


@dataclass
class Match:
    track: VideoTrack
    similarity: float          # best cosine between any of its embeddings and the subject's
    embeddings: int


def embed_track(reader: FrameReader, track: VideoTrack, det, identity: FaceIdentity,
                samples: int = SAMPLES) -> list[np.ndarray]:
    """SFace embeddings of the face in `track` at up to `samples` frames: the track box (padded)
    is cropped, the face detector finds the face and its landmarks inside, SFace aligns on them."""
    W, H = reader.width, reader.height
    f0, f1 = track.first_frame(), track.last_frame()
    frames = sorted({int(f0 + (f1 - f0) * k / max(1, samples - 1)) for k in range(samples)})
    out: list[np.ndarray] = []
    for f in frames:
        bb = track.bbox_at(f)
        if bb is None:
            continue
        x0, y0, x1, y1 = bb.to_pixels(W, H, CROP_PAD)
        if x1 - x0 < MIN_FACE_PX or y1 - y0 < MIN_FACE_PX:
            continue
        img = reader.read_at(f)
        if img is None:
            continue
        crop = img[y0:y1, x0:x1]
        faces = [d for d in det.detect(crop) if d.label == "face" and d.landmarks is not None]
        if not faces:
            continue
        face = max(faces, key=lambda d: d.bbox.w * d.bbox.h)
        e = identity.embed(crop, face)
        if e is not None:
            out.append(e)
    return out


def find_same_person(reader: FrameReader, subject: VideoTrack, candidates: list[VideoTrack], det,
                     identity: FaceIdentity, thr: float = SAME_PERSON, samples: int = SAMPLES,
                     progress: Optional[ProgressCB] = None, cancel: Optional[CancelCB] = None) -> tuple[list[Match], int]:
    """Matches among `candidates` for the person in `subject`. Returns (matches sorted by similarity,
    number of candidate tracks that could not be embedded at all)."""
    subj = embed_track(reader, subject, det, identity, samples)
    if not subj:
        raise RuntimeError("No usable face found in the selected track (too small or never detected).")
    S = np.stack(subj)
    matches: list[Match] = []
    unusable = 0
    for i, t in enumerate(candidates):
        if cancel and cancel():
            break
        if progress:
            progress(i / max(1, len(candidates)), f"Comparing track {i + 1} of {len(candidates)}")
        if t is subject or t.label != "face":
            continue
        embs = embed_track(reader, t, det, identity, samples)
        if not embs:
            unusable += 1
            continue
        best = float(np.max(np.stack(embs) @ S.T))
        if best >= thr:
            matches.append(Match(t, best, len(embs)))
    matches.sort(key=lambda m: -m.similarity)
    return matches, unusable
