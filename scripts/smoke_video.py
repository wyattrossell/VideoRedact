"""Track the red square in tests/data/sample.mp4, run auto-detect, export, and verify."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from videoredact.core.media import FrameReader, probe
from videoredact.core.model import Project, BBox, AudioRedaction, AudioStyle, VideoStyle
from videoredact.core.export import export_project, ExportOptions
from videoredact.vision.tracker import ObjectTracker
from videoredact.vision.auto_detect import run_auto_detect, AutoDetectOptions

src = "tests/data/sample.mp4"
info = probe(src, compute_hash=True)
proj = Project(media=info, author="smoke")
with FrameReader(src) as r:
    # red square at t=1s: x = -60 + (1/4)*760 = 130, y = 120+40*sin(2) ~ 156, size 60
    f = 30
    bb = BBox(128/640, 154/360, 64/640, 64/360)
    t0 = time.time()
    track = ObjectTracker(r).track(f, bb, label="manual", progress=lambda p, m: None)
    dt = time.time() - t0
    print(f"tracked 360 frames in {dt:.1f}s ({360/dt:.0f} fps); spans:",
          [(s.start_frame, s.end_frame, len(s.keyframes)) for s in track.spans])
    # Verify: at t=2s (frame 60) x should be ~320; at t=8s (frame 240) x = -60+ (2/6)*500 = 106
    for fr, ex in [(60, 320), (240, 106), (330, 356)]:
        b = track.bbox_at(fr)
        print(f"  frame {fr}: expected x~{ex}px got", None if b is None else f"{b.x*640:.0f}px (w {b.w*640:.0f})")
    assert track.bbox_at(150) is None, "square is off-screen at 5s; should be no box"
    assert track.bbox_at(60) is not None and track.bbox_at(240) is not None, "lost the square"
    proj.add_video_track(track)
    t0 = time.time()
    tracks = run_auto_detect(r, AutoDetectOptions(labels=["face", "screen", "document"], stride=10),
                             progress=lambda p, m: None)
    print(f"auto-detect ran in {time.time()-t0:.1f}s -> {len(tracks)} tracks (synthetic video: expect ~0)")
proj.add_audio_redaction(AudioRedaction(1.0, 2.5, text="test beep", style=AudioStyle.BEEP))
proj.add_audio_redaction(AudioRedaction(4.0, 5.0, text="test silence", style=AudioStyle.SILENCE))
proj.default_video_style = VideoStyle.BLACK  # solid colour: pixelate would leave a red square red
out = Path("tests/data/out/sample_redacted.mp4")
t0 = time.time()
res = export_project(proj, ExportOptions(str(out)), progress=lambda p, m: None)
print(f"export in {time.time()-t0:.1f}s:", {k: (v if not isinstance(v, str) or len(v) < 70 else v[:67] + '...') for k, v in res.items()})
oi = probe(str(out))
print("output:", oi.width, oi.height, oi.fps, oi.frame_count, "frames,", f"{oi.duration:.2f}s audio" if oi.has_audio else "NO AUDIO")
assert oi.frame_count == 360 and oi.has_audio
# check that the red square is gone at frame 60 in output but visible in input
with FrameReader(src) as a, FrameReader(str(out)) as b:
    fa, fb = a.read_at(60), b.read_at(60)
    red_in = int(((fa[:, :, 2] > 150) & (fa[:, :, 1] < 80)).sum())
    red_out = int(((fb[:, :, 2] > 150) & (fb[:, :, 1] < 80)).sum())
    print(f"red pixels frame 60: input {red_in}, output {red_out}")
    assert red_in > 2000 and red_out < red_in * 0.5
print("SMOKE OK")
