"""Regression: changing styles through the Qt combo boxes must not crash and must
round-trip through save/load. Qt returns str-based enums from itemData() as
plain str, which broke every style change in v0.1.3."""
import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from videoredact.core.model import (AudioRedaction, AudioStyle, BBox, Project, Shape, Span, VideoStyle,  # noqa: E402
                                    VideoTrack, MediaInfo, as_enum)
from videoredact.core import video_redact, audio_redact  # noqa: E402

import numpy as np  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_as_enum():
    assert as_enum(AudioStyle, "beep") is AudioStyle.BEEP
    assert as_enum(AudioStyle, AudioStyle.NOISE) is AudioStyle.NOISE
    assert as_enum(VideoStyle, None) is None
    with pytest.raises(ValueError):
        as_enum(Shape, "triangle")


def test_model_tolerates_plain_strings(tmp_path):
    p = Project(media=MediaInfo(path="x", fps=30, frame_count=10))
    p.default_audio_style = "silence"      # type: ignore[assignment]
    p.default_video_style = "blur"         # type: ignore[assignment]
    p.add_audio_redaction(AudioRedaction(0, 1, style="noise"))        # type: ignore[arg-type]
    t = VideoTrack(label="x", style="black", shape="ellipse")         # type: ignore[arg-type]
    t.spans = [Span(0, 5, {0: BBox(.1, .1, .3, .3)})]
    p.add_video_track(t)
    f = tmp_path / "s.vrproj"
    p.save(f)
    q = Project.load(f)
    assert q.default_audio_style is AudioStyle.SILENCE and q.video_tracks[0].shape is Shape.ELLIPSE
    frame = np.full((60, 60, 3), 200, np.uint8)
    video_redact.redact_frame(frame, p.video_tracks, 2, "black")      # type: ignore[arg-type]
    assert not np.array_equal(frame[8:22, 8:22], np.full((14, 14, 3), 200))
    x = np.zeros((8000, 1), np.float32)
    audio_redact.apply_redactions(x, 8000, p.audio_redactions, "beep")  # type: ignore[arg-type]


def test_main_window_style_changes(app, tmp_path):
    from videoredact.ui.main_window import MainWindow
    from videoredact.ui.dialogs import NewBoxDialog
    from videoredact.paths import settings
    w = MainWindow()
    w._load_media("tests/data/sample.mp4")
    app.processEvents()
    p = w.project
    p.add_audio_redaction(AudioRedaction(0.5, 1.0, text="x"))
    t = VideoTrack(label="screen", spans=[Span(0, 50, {0: BBox(.1, .1, .3, .3)})])
    p.add_video_track(t)
    w._project_changed()
    # toolbar default style combos (this crashed in 0.1.3)
    w.c_astyle.setCurrentIndex(1)
    w.c_vstyle.setCurrentIndex(2)
    assert p.default_audio_style is AudioStyle.SILENCE and p.default_video_style is VideoStyle.PIXELATE
    # per-item combos in the redactions panel
    row = w.panel.audio_root.child(0)
    combo = w.panel.tree.itemWidget(row, 4)
    combo.setCurrentIndex(4)  # Noise
    assert p.audio_redactions[0].style is AudioStyle.NOISE
    vrow = w.panel.video_root.child(0)
    vcombo = w.panel.tree.itemWidget(vrow, 4)
    vcombo.setCurrentIndex(2)  # Blur
    assert p.video_tracks[0].style is VideoStyle.BLUR
    # new-box dialog data comes back as plain values; the window coerces
    dlg = NewBoxDialog(settings, 1.0, 12.0)
    dlg.shape.setCurrentIndex(1)
    dlg.style.setCurrentIndex(3)
    assert as_enum(Shape, dlg.shape.currentData()) is Shape.ELLIPSE
    assert as_enum(VideoStyle, dlg.style.currentData()) is VideoStyle.PIXELATE
    # live tracking placeholder path must not raise before the job starts
    w._run_tracking(30, BBox(.2, .4, .1, .18), "manual", VideoTrack(label="manual", shape="ellipse"), step=3, backward=False)  # type: ignore[arg-type]
    assert any(tr.note.startswith("tracking in progress") for tr in p.video_tracks)
    w.jobs.cancel_all()
    w.jobs.wait_all(10000)
    app.processEvents()
    w._save_to(str(tmp_path / "ui.vrproj"))
    assert Project.load(tmp_path / "ui.vrproj").default_video_style is VideoStyle.PIXELATE
    w.close()
