"""Same-person matching (vision/people.py) and its bulk actions in the main window."""
import os
from pathlib import Path

import numpy as np
import pytest

from videoredact.core.model import BBox, Span, VideoTrack
from videoredact.vision import models as reg

SAMPLE = Path("tests/data/sample.mp4")


@pytest.mark.skipif(not (reg.is_available("yunet") and reg.is_available("sface")), reason="YuNet/SFace not downloaded")
def test_embed_and_match_on_a_real_frame(tmp_path):
    """Needs a frame with two faces (VR_TEST_FACE); footage is CJIS so nothing is committed."""
    import cv2
    p = os.environ.get("VR_TEST_FACE") or ""
    if not os.path.exists(p):
        pytest.skip("set VR_TEST_FACE=<image with two faces>")
    from videoredact.core.media import FrameReader
    from videoredact.vision.detector import CombinedDetector
    from videoredact.vision.identity import FaceIdentity
    from videoredact.vision import people
    img = cv2.imread(p)
    H, W = img.shape[:2]
    det = CombinedDetector(["face"], 0.5, 0.35, "yolox_tiny", 0, face_model="yunet")
    faces = det.detect(img)
    if len(faces) < 2:
        pytest.skip("test image needs two faces")
    # a one-frame "video": write the frame 10 times
    vid = tmp_path / "two_faces.mp4"
    w = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H))
    for _ in range(10):
        w.write(img)
    w.release()
    a, b = faces[0], faces[1]
    ta = VideoTrack(label="face", spans=[Span(0, 9, {0: a.bbox})])
    tb = VideoTrack(label="face", spans=[Span(0, 9, {0: b.bbox})])
    ta2 = VideoTrack(label="face", spans=[Span(2, 8, {2: a.bbox})])       # the same person again
    identity = FaceIdentity.try_create()
    with FrameReader(vid) as r:
        e = people.embed_track(r, ta, det, identity)
        assert e and all(abs(float(np.linalg.norm(x)) - 1) < 1e-3 for x in e)
        matches, unusable = people.find_same_person(r, ta, [tb, ta2], det, identity)
    assert [m.track.id for m in matches] == [ta2.id]
    assert matches[0].similarity > 0.8 and unusable == 0


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample clip missing (run scripts/make_sample.py)")
def test_person_actions_disable_delete_and_keep_only(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QMessageBox
    app = QApplication.instance() or QApplication([])
    from videoredact.ui.main_window import MainWindow
    from videoredact.vision.people import Match
    w = MainWindow()
    w._load_media(str(SAMPLE))
    app.processEvents()
    p = w.project
    subj = VideoTrack(label="face", spans=[Span(0, 30, {0: BBox(.1, .1, .1, .1)})])
    same1 = VideoTrack(label="face", spans=[Span(40, 60, {40: BBox(.2, .1, .1, .1)})])
    same2 = VideoTrack(label="face", spans=[Span(70, 90, {70: BBox(.3, .1, .1, .1)})])
    other = VideoTrack(label="face", spans=[Span(0, 90, {0: BBox(.6, .6, .1, .1)})])
    phone = VideoTrack(label="phone", spans=[Span(0, 90, {0: BBox(.8, .8, .1, .1)})])
    for t in (subj, same1, same2, other, phone):
        p.add_video_track(t)
    w._project_changed()
    matches = [Match(same1, 0.71, 3), Match(same2, 0.55, 2)]

    # "Keep only this person redacted": the other face is disabled, the phone untouched
    class Box:
        ActionRole = DestructiveRole = RejectRole = 0
        def __init__(self, *a, **k): pass
        def setWindowTitle(self, *a): pass
        def setText(self, *a): pass
        def setInformativeText(self, *a): pass
        buttons = {}
        def addButton(self, text, role):
            Box.buttons[text] = object()          # the window compares buttons by identity
            return Box.buttons[text]
        def exec(self): pass
        def clickedButton(self):
            return Box.buttons[Box.choice]
    real_box = __import__("videoredact.ui.main_window", fromlist=["QMessageBox"]).QMessageBox
    monkeypatch.setattr("videoredact.ui.main_window.QMessageBox", Box)
    Box.choice = "Keep only this person redacted"
    w._offer_person_actions(subj, matches, 0, 3)
    assert other.enabled is False and same1.enabled and same2.enabled and subj.enabled and phone.enabled
    assert "same person as track" in same1.note
    # "Disable them"
    other.enabled = True
    Box.choice = "Disable them"
    w._offer_person_actions(subj, matches, 0, 3)
    assert same1.enabled is False and same2.enabled is False and other.enabled
    # "Delete them" with undo
    Box.choice = "Delete them"
    w._offer_person_actions(subj, matches, 1, 3)
    assert p.get_track(same1.id) is None and p.get_track(same2.id) is None
    w.undo_delete()
    assert p.get_track(same2.id) is not None
    # no face selected: just a status message
    w._select("video", phone.id)
    w.find_same_person()
    # the close handler uses the real message box: put it back and answer Yes
    monkeypatch.setattr("videoredact.ui.main_window.QMessageBox", real_box)
    monkeypatch.setattr(real_box, "question", staticmethod(lambda *a, **k: real_box.Yes))
    monkeypatch.setattr(w, "_confirm_discard", lambda: True)
    w.close()
