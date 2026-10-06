"""Open the real GUI, load the sample, play ~2.5 s, verify frames/clock/stepping, then quit."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from videoredact.ui.main_window import MainWindow
from videoredact.core.model import AudioRedaction

app = QApplication(sys.argv)
w = MainWindow()
w.show()
w._load_media("tests/data/sample.mp4")
w.project.add_audio_redaction(AudioRedaction(0.5, 1.5, text="beep test"))
frames = []
w.video.frameShown.connect(frames.append)
errors = []
w.player.error.connect(errors.append)


def report():
    print("playing:", w.player.is_playing, "pos s:", round(w.player.position_s(), 2))
    print("frames shown:", len(frames), "last frame idx:", frames[-1] if frames else None)
    print("view size:", w.video.width_px, w.video.height_px, "pixmap null:", w.video.pix_item.pixmap().isNull())
    print("errors:", errors)
    pos = w.player.position_s()
    w.player.pause()
    p0 = w.player.current_frame()
    w.step_frames(1)
    p1 = w.player.current_frame()
    w.step_frames(-1)
    p2 = w.player.current_frame()
    w.seek(6.0)
    f6 = w.player.current_frame()
    print("step test:", p0, p1, p2, "seek 6s -> frame", f6)
    ok = len(frames) > 45 and not errors and 1.9 < pos < 3.5 and p1 == p0 + 1 and p2 == p0 and f6 == 180
    print("PLAYBACK OK" if ok else "PLAYBACK PROBLEM")
    app.quit()


QTimer.singleShot(800, w.toggle_play)
QTimer.singleShot(3300, report)
app.exec()
