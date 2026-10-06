"""Exercise the background jobs through the real main window (offscreen):
transcription (real whisper on the 12 s sample), tracking, auto-detect.
Fails loudly if any job errors or the UI does not receive results."""
import os
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PySide6.QtWidgets import QApplication, QMessageBox
from videoredact.ui.main_window import MainWindow
from videoredact.core.model import BBox, VideoTrack

# never block on message boxes in a smoke run
QMessageBox.information = staticmethod(lambda *a, **k: None)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
QMessageBox.warning = staticmethod(lambda *a, **k: None)

app = QApplication(sys.argv)
w = MainWindow()
w.show()
w._load_media("tests/data/sample.mp4")
app.processEvents()


def wait_jobs(timeout=300):
    t0 = time.time()
    while w.jobs.running and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.02)
    assert not w.jobs.running, "jobs still running after timeout"


wait_jobs()  # hashing
assert w.project.media.sha256, "hash job did not complete"

# --- transcription (real model) ---
t0 = time.time()
w.transcribe()
wait_jobs()
words = sum(len(s.words) for s in w.project.transcript)
print(f"transcription: {len(w.project.transcript)} segments, {words} words in {time.time()-t0:.1f}s; "
      f"transcript view words={len(w.transcript._words)}")
assert words >= 20 and len(w.transcript._words) == words
assert w.project.transcript_model

# --- tracking via the live-track path ---
proto = VideoTrack(label="manual")
t0 = time.time()
w._run_tracking(30, BBox(128 / 640, 154 / 360, 64 / 640, 64 / 360), "manual", proto, step=2, backward=True)
assert len(w.project.video_tracks) == 1 and w.project.video_tracks[0].note.startswith("tracking in progress")
wait_jobs()
tr = w.project.video_tracks[0]
print(f"tracking: spans {[(s.start_frame, s.end_frame) for s in tr.spans]} in {time.time()-t0:.1f}s; note='{tr.note}'")
assert tr.bbox_at(60) is not None and tr.bbox_at(240) is not None and tr.bbox_at(150) is None

# --- auto-detect on the synthetic clip (should run and find nothing) ---
from videoredact.vision.auto_detect import AutoDetectOptions, run_auto_detect
from videoredact.core.media import FrameReader
opts = AutoDetectOptions(labels=["face", "screen"], stride=10, obj_model="yolox_tiny")
w.jobs.run("Auto-detecting objects", lambda prog, cancel, partial: run_auto_detect(FrameReader("tests/data/sample.mp4"), opts, prog, cancel),
           lambda tracks: print("auto-detect tracks:", len(tracks)))
wait_jobs()

# --- cancel path ---
w.jobs.run("Slow job", lambda prog, cancel, partial: [time.sleep(0.05) for _ in range(200) if not cancel()],
           lambda r: None, on_cancel=lambda: print("cancel callback fired"))
app.processEvents()
w.jobs.cancel_all()
wait_jobs()
print("JOBS SMOKE OK")
