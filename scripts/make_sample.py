"""Generate synthetic sample media for tests and manual QA.

Produces tests/data/sample.mp4: 12 s, 640x360 @ 30 fps. A red square moves
left->right, leaves the frame (4-6 s), re-enters from the left and continues.
A blue rectangle stays static. Audio is Windows text-to-speech reading a
script with names and a phone number so transcript/PII features can be
exercised. Without System.Speech (non-Windows) a 440 Hz tone is used.
"""
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from videoredact.core.media import ffmpeg_exe  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "tests" / "data"
OUT.mkdir(parents=True, exist_ok=True)
SCRIPT = ("Officer Johnson speaking. The suspect gave his name as Michael Carter. "
          "His phone number is five five five, two one two, four four six seven. "
          "Michael Carter was last seen on Maple Street. Johnson, clear.")
W, H, FPS, DUR = 640, 360, 30, 12


def square_x(t: float) -> float:
    """Red square x position (px). Off-screen between 4 s and 6 s."""
    if t < 4:
        return -60 + (t / 4) * 760
    if t < 6:
        return -200
    return -60 + ((t - 6) / 6) * 500


def tts(path: Path) -> bool:
    ps = f"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.Rate = 1
$s.SetOutputToWaveFile('{path}')
$s.Speak('{SCRIPT}')
$s.Dispose()
"""
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, timeout=120)
        return r.returncode == 0 and path.exists() and path.stat().st_size > 10000
    except Exception:
        return False


def main():
    wav = OUT / "speech.wav"
    have_tts = tts(wav)
    ff = ffmpeg_exe()
    out = OUT / "sample.mp4"
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "pipe:0"]
    if have_tts:
        cmd += ["-i", str(wav)]
    else:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={DUR}"]
    cmd += ["-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-t", str(DUR), str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for i in range(DUR * FPS):
        t = i / FPS
        frame = np.full((H, W, 3), 64, dtype=np.uint8)
        # textured background so trackers have something to lock onto besides the square
        cv2.rectangle(frame, (40, 250), (240, 330), (200, 60, 30), -1)      # static blue-ish box
        cv2.circle(frame, (520, 90), 35, (90, 180, 90), -1)                 # static green circle
        x = int(square_x(t))
        y = int(120 + 40 * math.sin(t * 2))
        cv2.rectangle(frame, (x, y), (x + 60, y + 60), (0, 0, 230), -1)      # moving red square
        cv2.putText(frame, f"{t:05.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    err = proc.stderr.read().decode("utf-8", "replace")
    if proc.wait() != 0:
        print(err)
        sys.exit(1)
    print("wrote", out, "(tts)" if have_tts else "(tone)")


if __name__ == "__main__":
    main()
