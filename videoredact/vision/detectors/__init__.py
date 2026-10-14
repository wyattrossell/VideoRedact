"""Detector registry: build a detector pipeline from a config dict.

    det = build({"face": "yunet", "tiles": True, "head_fallback": True,
                 "objects": ["screen", "document", "phone", "person"], "face_pad": 0.25})
    det.detect([frame1, frame2]) -> [[Detection,...], [...]]

Face backends: yunet (default, MIT), centerface (MIT), scrfd (eval only), egoblur (eval only, torch).
Object backend: yolox_tiny / yolox_s (Apache). Open-vocabulary: see openvocab.py (eval only).
"""
from __future__ import annotations

from typing import Any, Optional

from .base import Detection, Detector, PerFrameDetector, nms_merge
from .wrappers import EnsembleDetector, HeadFallback, MultiScaleDetector, TiledDetector

FACE_BACKENDS = ("yunet", "centerface", "scrfd", "egoblur")


def build_face(backend: str = "yunet", score_thr: float = 0.5, max_side: int = 1280, tiles: bool = False,
               tile: int = 640, multiscale: Optional[list[float]] = None, threads: int = 0) -> Detector:
    from . import adapters
    if backend == "yunet":
        det: Detector = adapters.YuNetDetector(score_thr, max_side=max_side)
    elif backend == "centerface":
        det = adapters.CenterFaceDetector(min(score_thr, 0.5), max_side=max_side, threads=threads)
    elif backend == "scrfd":
        from .scrfd import SCRFDDetector
        det = SCRFDDetector(score_thr=score_thr, input_size=640, threads=threads)
    elif backend == "egoblur":
        det = adapters.EgoBlurDetector("face", score_thr=score_thr, max_side=max_side)
    else:
        raise ValueError(f"unknown face backend {backend}")
    if multiscale:
        det = MultiScaleDetector(det, multiscale)
    if tiles:
        det = TiledDetector(det, tile=tile, min_side=1000)
    return det


def build(cfg: dict[str, Any]) -> Detector:
    members: list[Detector] = []
    threads = int(cfg.get("threads", 0))
    if cfg.get("face", "yunet"):
        members.append(build_face(cfg.get("face", "yunet"), float(cfg.get("face_conf", 0.5)),
                                  int(cfg.get("face_max_side", 1280)), bool(cfg.get("tiles", False)),
                                  int(cfg.get("tile", 640)), cfg.get("multiscale"), threads))
    objects = set(cfg.get("objects") or [])
    if cfg.get("head_fallback") and "person" not in objects:
        objects.add("person")
    if objects:
        from . import adapters
        members.append(adapters.YOLOXDetector(cfg.get("object_model", "yolox_tiny"), float(cfg.get("obj_conf", 0.35)),
                                              labels=objects, threads=threads))
    if cfg.get("open_vocab"):
        from .openvocab import GroundingDinoDetector
        members.append(GroundingDinoDetector(prompts=cfg["open_vocab"], score_thr=float(cfg.get("ov_conf", 0.3))))
    det: Detector = members[0] if len(members) == 1 else EnsembleDetector(members)
    if cfg.get("head_fallback") or float(cfg.get("face_pad", 0)) > 0:
        det = HeadFallback(det, face_pad=float(cfg.get("face_pad", 0)))
    return det


__all__ = ["Detection", "Detector", "PerFrameDetector", "nms_merge", "EnsembleDetector", "HeadFallback",
           "MultiScaleDetector", "TiledDetector", "build", "build_face", "FACE_BACKENDS"]
