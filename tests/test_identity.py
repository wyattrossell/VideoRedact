"""Face landmarks from the detectors and SFace identity embeddings (skipped without the models)."""
import numpy as np
import pytest

from videoredact.vision import models as reg


def _face_image():
    """A real face is needed (YuNet ignores drawn ones) and none is committed: footage is CJIS.
    Point VR_TEST_FACE at any local image with a face to run these tests."""
    import os
    import cv2
    p = os.environ.get("VR_TEST_FACE") or "tests/data/face.jpg"
    if not os.path.exists(p):
        pytest.skip("no face test image (set VR_TEST_FACE=<image with a face>)")
    return cv2.imread(p)


@pytest.mark.skipif(not reg.is_available("yunet"), reason="YuNet not downloaded")
def test_detectors_report_landmarks_in_frame_coordinates():
    from videoredact.vision.detector import CombinedDetector
    img = _face_image()
    H, W = img.shape[:2]
    det = CombinedDetector(["face"], 0.5, 0.35, "yolox_tiny", 0, face_model="yunet")
    faces = det.detect(img)
    assert faces, "expected a face in tests/data/face.jpg"
    f = faces[0]
    assert f.landmarks is not None and f.landmarks.shape == (5, 2)
    assert np.all(f.landmarks >= 0) and np.all(f.landmarks <= 1)
    # the landmarks lie inside (or just around) the box
    b = f.bbox
    for x, y in f.landmarks:
        assert b.x - 0.1 * b.w <= x <= b.x + 1.1 * b.w and b.y - 0.1 * b.h <= y <= b.y + 1.1 * b.h
    # the OpenCV DNN backend agrees
    det_cv = CombinedDetector(["face"], 0.5, 0.35, "yolox_tiny", 0, face_model="yunet_cv")
    g = det_cv.detect(img)[0]
    assert g.landmarks is not None and float(np.abs(g.landmarks - f.landmarks).max()) < 0.02


@pytest.mark.skipif(not (reg.is_available("yunet") and reg.is_available("sface")), reason="YuNet/SFace not downloaded")
def test_identity_embedding_is_unit_length_and_self_similar():
    import cv2
    from videoredact.vision.detector import CombinedDetector
    from videoredact.vision.identity import FaceIdentity
    img = _face_image()
    fi = FaceIdentity.try_create()
    assert fi is not None
    det = CombinedDetector(["face"], 0.5, 0.35, "yolox_tiny", 0, face_model="yunet")
    f = det.detect(img)[0]
    e1 = fi.embed(img, f)
    assert e1 is not None and abs(float(np.linalg.norm(e1)) - 1.0) < 1e-3
    # same face, slightly different pixels (blur) stays the same person
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    f2 = det.detect(blurred)[0]
    e2 = fi.embed(blurred, f2)
    assert FaceIdentity.cosine(e1, e2) > 0.7
    # no landmarks -> no embedding, no crash
    f.landmarks = None
    assert fi.embed(img, f) is None


def test_identity_unavailable_degrades_gracefully(monkeypatch):
    from videoredact.vision import identity
    monkeypatch.setattr(reg, "is_available", lambda name: False)
    assert identity.FaceIdentity.try_create() is None
