"""GUI-level check on real body-cam footage (offscreen window):
draw-box -> live tracking job -> style change -> export -> verify the box is
actually black in the output. Uses samples/ (not in git). ~4-5 minutes.

Usage: python scripts/smoke_realvideo.py [path-to-video]
"""
import os
import sys
import time
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QMessageBox
from videoredact.core.media import FrameReader, probe
from videoredact.core.model import BBox, VideoStyle, VideoTrack
from videoredact.core.export import ExportOptions
from videoredact.ui.main_window import MainWindow

SRC = sys.argv[1] if len(sys.argv) > 1 else "samples/10-04-2026 Traffic Stop N Preston Hwy and Lodi Ln .MP4"
QMessageBox.information = staticmethod(lambda *a, **k: None)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)

app = QApplication(sys.argv)
w = MainWindow()
w.show()
w._load_media(SRC)
app.processEvents()
info = w.project.media
print(f"loaded {Path(SRC).name}: {info.width}x{info.height} {info.fps:.2f} fps {info.frame_count} frames")


def wait_jobs(timeout=900):
    t0 = time.time()
    while w.jobs.running and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.02)
    assert not w.jobs.running, "jobs still running"


wait_jobs()

# 1. "draw a box" around the laptop screen at frame 5064 and track 60 s forward + backward
start = 5064
bb = BBox(643 / 1280, 95 / 720, 390 / 1280, 605 / 720)
t0 = time.time()
w._run_tracking(start, bb, "screen", VideoTrack(label="screen"), step=2, backward=True, end_frame=start + 1800)
app.processEvents()
assert w.project.video_tracks and w.project.video_tracks[0].note.startswith("tracking in progress")
wait_jobs()
tr = w.project.video_tracks[0]
print(f"tracked in {time.time()-t0:.0f}s: spans={[(s.start_frame, s.end_frame) for s in tr.spans]} note='{tr.note}'")
assert tr.bbox_at(start + 900) is not None, "track lost the screen"

# 2. change the default style via the toolbar (crashed in 0.1.3) and the item style via the panel
w.c_vstyle.setCurrentIndex(0)      # black
w.c_astyle.setCurrentIndex(1)      # silence
vrow = w.panel.video_root.child(0)
w.panel.tree.itemWidget(vrow, 4).setCurrentIndex(0)  # default
assert tr.style is None and w.project.default_video_style is VideoStyle.BLACK

# 3. preview path renders the box on a real frame
with FrameReader(SRC) as r:
    frame = r.read_at(start + 600)
w.video.set_tracks(w.project.video_tracks, w.project.default_video_style)
w.video.show_still(frame, start + 600)
app.processEvents()
assert len(w.video._boxes) == 1, "preview did not draw the tracked box"

# 4. export the whole file and verify the box region is black inside the span, untouched outside
out = Path(os.environ.get("TEMP", ".")) / "videoredact_realvideo_export.mp4"
opts = ExportOptions(str(out), crf=28, preset="ultrafast", write_report=True, write_project=False)
from videoredact.core.export import export_project
t0 = time.time()
res = {}
w.jobs.run("Export", lambda prog, cancel, partial: export_project(w.project, opts, prog, cancel), lambda r_: res.update(r_))
wait_jobs()
print(f"export in {time.time()-t0:.0f}s -> {res.get('output')} sha={res.get('output_sha256', '')[:12]}")
with FrameReader(SRC) as a, FrameReader(str(out)) as b:
    f_in, f_out = a.read_at(start + 600), b.read_at(start + 600)
    box = tr.bbox_at(start + 600)
    x0, y0, x1, y1 = box.to_pixels(1280, 720, pad=0.0)
    inner_out = f_out[y0 + 10:y1 - 10, x0 + 10:x1 - 10]
    inner_in = f_in[y0 + 10:y1 - 10, x0 + 10:x1 - 10]
    print(f"inside box: input mean {inner_in.mean():.0f}, output mean {inner_out.mean():.1f} (black expected)")
    assert inner_out.mean() < 8 and inner_in.mean() > 20
    f_in2, f_out2 = a.read_at(100), b.read_at(100)   # before the span: should be the same picture
    diff = np.abs(f_in2.astype(int) - f_out2.astype(int)).mean()
    print(f"outside span: mean abs diff {diff:.2f} (compression noise only)")
    assert diff < 6
print("REAL VIDEO SMOKE OK")
