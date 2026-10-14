"""Multi-object tracking adapter over Roboflow `trackers` (Apache-2.0).

Wraps BoT-SORT (default; camera-motion compensation for moving body cams),
ByteTrack and OC-SORT behind one call:

    mot = MOT("botsort", fps=30, stride=5)
    tracked = mot.update(detections, frame, frame_idx)   # -> list[TrackedDet]

Each TrackedDet carries a stable `track_id`. Detections are fed every
`stride` frames; the Kalman filters are told the real timestamp so motion
prediction spans the gap correctly. Classes are tracked together (an id can
not change class: the label of a track is its first detection's label).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from videoredact.core.model import BBox
from .detectors.base import Detection


@dataclass
class TrackedDet:
    track_id: int
    bbox: BBox
    label: str
    score: float


class MOT:
    def __init__(self, kind: str = "botsort", fps: float = 30.0, stride: int = 1, lost_buffer_steps: int = 8,
                 activation_thr: float = 0.3, min_consecutive: int = 1, cmc: bool = True):
        import trackers
        self.kind = kind
        eff_fps = max(1.0, fps / max(1, stride))
        if kind == "botsort":
            self.tr = trackers.BoTSORTTracker(lost_track_buffer=lost_buffer_steps, frame_rate=eff_fps,
                                              track_activation_threshold=activation_thr,
                                              minimum_consecutive_frames=min_consecutive, enable_cmc=cmc,
                                              cmc_method="sparseOptFlow", cmc_downscale=2,
                                              instant_first_frame_activation=True)
        elif kind == "bytetrack":
            self.tr = trackers.ByteTrackTracker(lost_track_buffer=lost_buffer_steps, frame_rate=eff_fps,
                                                track_activation_threshold=activation_thr,
                                                minimum_consecutive_frames=min_consecutive)
        elif kind == "ocsort":
            self.tr = trackers.OCSORTTracker(lost_track_buffer=lost_buffer_steps, frame_rate=eff_fps,
                                             minimum_consecutive_frames=min_consecutive)
        else:
            raise ValueError(kind)
        self.fps = fps
        self._labels: dict[int, str] = {}
        self._label_ids: dict[str, int] = {}

    def update(self, dets: Sequence[Detection], frame: Optional[np.ndarray], frame_idx: int) -> list[TrackedDet]:
        import supervision as sv
        H, W = (frame.shape[:2] if frame is not None else (1, 1))
        if not dets:
            xyxy = np.zeros((0, 4), dtype=np.float32)
            conf = np.zeros((0,), dtype=np.float32)
            cls = np.zeros((0,), dtype=int)
        else:
            xyxy = np.array([[d.bbox.x * W, d.bbox.y * H, (d.bbox.x + d.bbox.w) * W, (d.bbox.y + d.bbox.h) * H] for d in dets], dtype=np.float32)
            conf = np.array([d.score for d in dets], dtype=np.float32)
            cls = np.array([self._label_ids.setdefault(d.label, len(self._label_ids)) for d in dets], dtype=int)
        sd = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)
        out = self.tr.update(sd, frame=frame, timestamp=frame_idx / self.fps)
        res: list[TrackedDet] = []
        ids = out.tracker_id if out.tracker_id is not None else np.full(len(out), -1)
        inv = {v: k for k, v in self._label_ids.items()}
        for i in range(len(out)):
            tid = int(ids[i])
            if tid < 0:
                continue
            x1, y1, x2, y2 = out.xyxy[i]
            label = inv.get(int(out.class_id[i]), "object") if out.class_id is not None else "object"
            label = self._labels.setdefault(tid, label)
            score = float(out.confidence[i]) if out.confidence is not None else 1.0
            res.append(TrackedDet(tid, BBox.from_pixels(x1, y1, x2, y2, W, H), label, score))
        return res
