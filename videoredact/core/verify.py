"""Post-export verification: re-detect faces (and optionally other classes) in
the *redacted output* and report any detection not covered by a redaction.

Why: "redaction is only as good as one missed frame". An independent second
pass over the released file catches tracker drift, detector gaps between
strides, and faces that were never detected the first time. The result is a
list of (time, label, box) findings the reviewer can jump to, plus a coverage
figure for the report. Findings are advisory; the reviewer decides.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import cv2

from .media import FrameReader
from .model import BBox, Project
from .video_redact import boxes_at

ProgressCB = Callable[[float, str], None]
CancelCB = Callable[[], bool]


@dataclass
class Finding:
    frame: int
    time: float
    label: str
    bbox: BBox
    score: float
    covered: float   # fraction of the detection covered by redactions (0..1)


@dataclass
class VerifyResult:
    frames_checked: int
    detections: int
    uncovered: list[Finding]
    labels: list[str]

    @property
    def ok(self) -> bool:
        return not self.uncovered


def _coverage(det: BBox, boxes: list[BBox]) -> float:
    """Approximate fraction of `det` covered by the union of `boxes` (grid sampling)."""
    if not boxes:
        return 0.0
    n = 6
    hit = 0
    for i in range(n):
        for j in range(n):
            px = det.x + (i + 0.5) / n * det.w
            py = det.y + (j + 0.5) / n * det.h
            if any(b.x <= px <= b.x + b.w and b.y <= py <= b.y + b.h for b in boxes):
                hit += 1
    return hit / (n * n)


def verify_output(output_path: str, project: Project, labels: Optional[list[str]] = None,
                  stride: int = 15, min_cover: float = 0.6, face_conf: float = 0.5,
                  progress: Optional[ProgressCB] = None, cancel: Optional[CancelCB] = None,
                  frame_offset: int = 0) -> VerifyResult:
    """Scan the exported file every `stride` frames. A detection counts as
    uncovered when less than `min_cover` of it lies inside enabled redaction
    boxes for that frame (padding included). Detectors run on the redacted
    output, so anything they still find is by definition visible."""
    from videoredact.vision.detector import CombinedDetector
    labels = labels or ["face"]
    det = CombinedDetector(labels, face_conf=face_conf)
    tracks = [t for t in project.video_tracks if t.enabled]
    found: list[Finding] = []
    n_det = 0
    checked = 0
    with FrameReader(output_path) as r:
        W, H = r.width, r.height
        total = r.frame_count or 1
        fps = r.fps or project.media.fps or 30.0
        idx = 0
        r.seek(0)
        while True:
            if cancel and cancel():
                break
            frame = r.read()
            if frame is None:
                break
            if idx % stride == 0:
                checked += 1
                dets = det.detect(frame)
                if dets:
                    red = []
                    for t, bb in boxes_at(tracks, idx + frame_offset):
                        x0, y0, x1, y1 = bb.to_pixels(W, H, t.pad)
                        red.append(BBox.from_pixels(x0, y0, x1, y1, W, H))
                    for d in dets:
                        n_det += 1
                        cov = _coverage(d.bbox, red)
                        if cov < min_cover:
                            src_idx = idx + frame_offset
                            found.append(Finding(src_idx, src_idx / fps, d.label, d.bbox, d.score, cov))
                if progress and checked % 10 == 0:
                    progress(min(0.999, idx / total), f"Verifying frame {idx}/{total}: {len(found)} uncovered so far")
            idx += 1
    if progress:
        progress(1.0, f"Verification complete: {len(found)} uncovered detection(s)")
    return VerifyResult(checked, n_det, found, labels)
