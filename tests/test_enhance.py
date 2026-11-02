"""Detector-side brightness/contrast boost (vision/enhance.py): the detectors see an enhanced
copy; the picture the reviewer sees and the exported file never change."""
import hashlib
import os

import numpy as np
import pytest

from videoredact.vision.enhance import enhance_for_detection, luma_stats, stretch_lut


def _scene(level: float, spread: float, seed: int = 0) -> np.ndarray:
    """A synthetic 720p frame: a textured background at `level` with a `spread` of grey levels
    and a brighter square (a 'face') in one corner."""
    rng = np.random.default_rng(seed)
    base = level + (rng.random((360, 640)) - 0.5) * spread
    img = np.repeat(np.clip(base, 0, 255)[..., None], 3, axis=2)
    img[200:260, 500:560] = np.clip(level + spread, 0, 255)
    return img.astype(np.uint8)


def test_dark_flat_frame_gets_brighter_and_more_contrast_without_touching_the_input():
    dark = _scene(18, 12)
    before = dark.copy()
    out = enhance_for_detection(dark)
    assert np.array_equal(dark, before)                       # input untouched
    assert out.shape == dark.shape and out.dtype == np.uint8
    m0, s0 = luma_stats(dark)
    m1, s1 = luma_stats(out)
    assert m1 > m0 * 3 and s1 > s0 * 3                         # lifted and spread out


def test_well_exposed_frame_is_left_nearly_alone():
    good = _scene(120, 200)
    out = enhance_for_detection(good)
    m0, s0 = luma_stats(good)
    m1, s1 = luma_stats(out)
    assert abs(m1 - m0) < 20 and 0.7 * s0 < s1 < 1.5 * s0
    # a frame already spanning the whole range gets no stretch LUT at all
    full = np.repeat(np.linspace(0, 255, 640, dtype=np.uint8)[None, :, None], 360, axis=0)
    assert stretch_lut(np.repeat(full, 3, axis=2)[:, :, 0]) is None


def test_flat_or_odd_frames_do_not_crash():
    flat = np.full((64, 64, 3), 77, np.uint8)
    assert enhance_for_detection(flat).shape == flat.shape
    grey = np.zeros((64, 64), np.uint8)
    assert enhance_for_detection(grey) is grey                 # not BGR: returned as is


def test_auto_mode_boosts_only_dark_or_flat_frames():
    from videoredact.vision.enhance import as_mode, boost, needs_boost
    dark, flat, good = _scene(18, 12), _scene(120, 20), _scene(120, 200)
    assert needs_boost(dark) and needs_boost(flat) and not needs_boost(good)
    assert boost(good, "auto") is good and boost(good, "off") is good and boost(good, "always") is not good
    assert boost(dark, "auto") is not dark and boost(dark, "off") is dark
    assert as_mode(True) == "always" and as_mode(False) == "off" and as_mode(None) == "off"
    assert as_mode("Always") == "always" and as_mode("nonsense") == "auto" and as_mode("auto") == "auto"


def test_pipeline_boosts_faces_and_plates_only():
    """The wrapper sits innermost in the face pipeline; YOLOX members see the original."""
    from videoredact.vision.detectors import build, Enhanced, EnsembleDetector, MultiScaleDetector
    from videoredact.vision.detectors.wrappers import HeadFallback
    from videoredact.vision import models as reg
    if not reg.is_available("yunet") or not reg.is_available("yolox_tiny"):
        pytest.skip("models not installed")
    det = build({"face": "yunet", "multiscale": [1.0, 1.5], "objects": ["screen"], "face_pad": 0.25, "enhance": "always"})
    assert isinstance(det, HeadFallback) and isinstance(det.inner, EnsembleDetector)
    face, yolox = det.inner.members
    assert isinstance(face, MultiScaleDetector) and isinstance(face.inner, Enhanced) and face.inner.mode == "always"
    assert not isinstance(yolox, Enhanced) and yolox.name == "yolox_tiny"
    plain = build({"face": "yunet", "multiscale": [1.0, 1.5], "objects": ["screen"], "face_pad": 0.25, "enhance": "off"})
    assert not isinstance(plain.inner.members[0].inner, Enhanced)
    auto = build({"face": "yunet", "multiscale": None, "objects": [], "face_pad": 0, "enhance": "auto"})
    assert isinstance(auto, Enhanced) and auto.mode == "auto"
    # the plate detector carries the mode into its full-frame and crop passes
    if reg.is_available("plate_yolov9t"):
        vp = build({"face": "", "objects": ["license_plate"], "face_pad": 0, "enhance": "auto"})
        assert vp.plate_det.enhance == "auto"


@pytest.mark.skipif(not os.path.exists("tests/data/sample.mp4"), reason="sample clip missing")
def test_export_pixels_do_not_depend_on_the_detector_option(tmp_path):
    """The option only changes what the detectors are shown. Two exports of the same project,
    with the option on and off, must be byte-identical."""
    from videoredact.core.export import ExportOptions, export_project
    from videoredact.core.media import probe
    from videoredact.core.model import BBox, Project, Span, VideoTrack
    proj = Project(media=probe("tests/data/sample.mp4"))
    proj.add_video_track(VideoTrack(label="face", spans=[Span(5, 40, {5: BBox(.2, .2, .2, .3), 40: BBox(.5, .3, .2, .3)})]))
    outs = []
    for enh in ("off", "always"):
        out = tmp_path / f"e_{enh}.mp4"
        export_project(proj, ExportOptions(str(out), write_report=False, write_project=False, write_audit=False,
                                           verify_labels=["face"], verify_enhance=enh))
        outs.append(hashlib.sha256(out.read_bytes()).hexdigest())
    assert outs[0] == outs[1]


@pytest.mark.skipif(not os.path.exists("tests/data/sample.mp4"), reason="sample clip missing")
def test_auto_detect_and_tracking_run_with_the_option_on():
    from videoredact.core.media import FrameReader
    from videoredact.core.model import BBox
    from videoredact.vision import models as reg
    from videoredact.vision.auto_detect import AutoDetectOptions, run_auto_detect
    from videoredact.vision.tracker import ObjectTracker, TrackOptions
    if not reg.is_available("yunet"):
        pytest.skip("face model not installed")
    with FrameReader("tests/data/sample.mp4") as r:
        tracks = run_auto_detect(r, AutoDetectOptions(labels=["face"], stride=10, workers=1, end_frame=60, enhance="always"))
        assert isinstance(tracks, list)
        t = ObjectTracker(r, None, TrackOptions(engine="anchored-all", step=2, end_frame=30, backward=False, enhance="always")
                          ).track(5, BBox(.3, .3, .2, .3), label="other")
        assert t is not None
