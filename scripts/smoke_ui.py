"""Construct the main window offscreen, load the sample, exercise key operations."""
import os, sys, time
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("QT_MEDIA_BACKEND", "windows")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from videoredact.ui.main_window import MainWindow
from videoredact.core.model import BBox, AudioRedaction, Project
from videoredact.audio.transcribe import find_matches

app = QApplication(sys.argv)
w = MainWindow()
w.show()
w._load_media("tests/data/sample.mp4")
app.processEvents()
assert w.project is not None and w.project.media.has_video
print("loaded:", w.project.media.width, w.project.media.height, w.project.media.fps)

# fake a transcript so word redaction can be exercised without running whisper
from videoredact.core.model import Segment, Word
w.project.transcript = [Segment(0, 2, "Officer Johnson speaking", [Word(0.0, 0.4, "Officer"), Word(0.46, 0.8, "Johnson"), Word(0.9, 1.5, "speaking.")]),
                        Segment(3, 5, "Johnson clear", [Word(3.0, 3.4, "Johnson,"), Word(3.5, 4.0, "clear.")])]
w.transcript.rebuild(); w._update_enabled()
w.redact_all_of("johnson")
assert len(w.project.audio_redactions) == 2, w.project.audio_redactions
w.transcript.search.setText("johnson"); app.processEvents()
print("search label:", w.transcript.count_lbl.text())
assert "2 match" in w.transcript.count_lbl.text()

# add a static video box and a keyframe track, edit a box
from videoredact.core.model import VideoTrack, Span
t = VideoTrack(label="screen", spans=[Span(0, 100, {0: BBox(.1, .1, .2, .2)})])
w.project.add_video_track(t); w._project_changed()
w.seek(1.0); app.processEvents()
w.on_box_edited(t.id, BBox(.3, .3, .2, .2))
assert t.bbox_at(30) is not None and abs(t.bbox_at(30).x - .3) < 1e-6
print("panel rows:", w.panel.audio_root.childCount(), "audio,", w.panel.video_root.childCount(), "video")
assert w.panel.video_root.childCount() == 1 and w.panel.audio_root.childCount() == 2

# render a frame through the view directly (sink may not deliver offscreen)
import cv2
from videoredact.core.media import FrameReader
with FrameReader("tests/data/sample.mp4") as r:
    fr = r.read_at(30)
w.video.set_tracks(w.project.video_tracks, w.project.default_video_style)
w.video.show_still(fr, 30)
app.processEvents()
assert not w.video.pix_item.pixmap().isNull()
print("boxes drawn on view:", len(w.video._boxes))
assert len(w.video._boxes) == 1

# delete via panel + timeline refresh
w._delete("video", t.id)
assert not w.project.video_tracks
# save & reload
out = Path("tests/data/out/ui_smoke.vrproj"); out.parent.mkdir(exist_ok=True)
w._save_to(str(out))
p2 = Project.load(out)
assert len(p2.audio_redactions) == 2 and len(p2.audit_log) >= 5
print("audit entries:", len(p2.audit_log))

# dialogs construct
from videoredact.ui.dialogs import SettingsDialog, ExportDialog, AutoDetectDialog, NewBoxDialog
from videoredact.paths import settings
for d in (SettingsDialog(settings, w.project), ExportDialog(w.project, settings), AutoDetectDialog(settings, False), NewBoxDialog(settings, 1.0, 12.0)):
    d.show(); app.processEvents(); d.close()
print("UI SMOKE OK")
QTimer.singleShot(0, app.quit); app.exec()
