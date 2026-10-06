# Developer notes (read this first when switching machines)

Living log for Wyatt + Claude. Newest entries at the top of each section.
Keep it short; details belong in code comments or docs/.

## Quick start on a new machine

```powershell
git clone https://github.com/wyattrossell/VideoRedact.git
cd VideoRedact
.\scripts\setup_dev.ps1        # python.org Python 3.11+; makes C:\dev\venv-videoredact
.\scripts\run_dev.ps1          # launches the app
C:\dev\venv-videoredact\Scripts\python -m pytest
C:\dev\venv-videoredact\Scripts\python scripts\make_sample.py   # regenerates tests/data/sample.mp4 (needs Windows TTS)
```

Why a venv at `C:\dev\...`: the Microsoft Store Python puts site-packages under
a very deep path and PySide6 then fails to install with "No such file or
directory" (Windows 260-char limit). Also do not put the venv on the
redirected Desktop share; it is slow and triggers git "dubious ownership"
(fixed once per machine with `git config --global --add safe.directory <path>`).

## Current state (2026-10-06)

Working end to end, verified by scripts/smoke_*.py:
- Transcription: faster-whisper `small` int8 on CPU, 12 s clip in 3.4 s (3.5x real time).
- Word/phrase matching across transcript, rule-based PII suggestions.
- Single-object tracking (CSRT at <=640 px) with loss detection and
  re-acquisition via class detector + multi-scale template matching; tracks
  backward from the drawn box as well. Synthetic test: square leaves at 3.6 s,
  re-acquired at 6.7 s, position error ~2 px.
- Auto-detect (YuNet faces + YOLOX objects every N frames, IoU linking).
- Export: OpenCV -> raw BGR pipe -> libx264 + AAC; audio redaction with
  crossfades outside the hard range; CSV + PDF report; .vrproj sidecar.
- UI: custom preview player (see below), transcript with right-click redact,
  timeline lanes, redactions panel with per-item style, settings, export dialog.

Not yet done: see docs/ROADMAP.md. Biggest gaps: license-plate detector,
undo/redo, real-footage QA, installer build has not been run yet.

## Decisions

- **Permissive licensing only** (MIT/Apache/BSD/LGPL-dynamic). No Ultralytics
  (AGPL), no boxmot (AGPL), no StrongSORT (GPL). ByteTrack-style association is
  our own implementation in vision/auto_detect.py.
- **Target machine**: 8 cores / 16 GB, no GPU. Everything on CPU, ONNX Runtime
  + OpenCV. Detection runs on frames downscaled to <=960 px, tracking <=640 px.
- **Redaction styles are user-selectable** per project default and per item:
  audio beep/silence/low tone/noise; video black/blur/pixelate; rect/ellipse.
  Black box is the only style we describe as guaranteed irreversible.
- **Reports**: PDF + CSV with SHA-256 of source and output and the audit log; no
  thumbnails of redacted content. Project file is JSON (`.vrproj`).
- **Boxes stored normalized (0..1)** with keyframes per visibility span;
  interpolation between keyframes; spans are simplified on save.
- **Preview player is our own** (ui/player.py): OpenCV frames + FFmpeg PCM
  pipe into QAudioSink. Reason: QMediaPlayer (both "ffmpeg" and "windows"
  backends, PySide6 6.11.2) stalled at position 0 whenever a window was
  visible and an audio output was attached, on the dev laptop (Win 11,
  Realtek audio). QAudioSink works fine. Side benefit: the preview plays the
  actual beep/silence, and frame stepping is exact.
- **Models dir**: `%LOCALAPPDATA%\VideoRedact\models` (or bundled `models\`
  when frozen; override `VIDEOREDACT_MODELS`). Whisper models go in
  `models\whisper` via faster-whisper's `download_root`.

## Gotchas

- Store Python virtualizes `%LOCALAPPDATA%`; HF cache warnings about symlinks
  are harmless (set HF_HUB_DISABLE_SYMLINKS_WARNING=1, done in app.py).
- ffmpeg `drawbox` does not evaluate `t` per frame in 7.1; the sample
  generator draws frames with OpenCV and pipes them to ffmpeg instead.
- Pixelating a solid-colour object leaves it the same colour (by design);
  tests for redaction presence must use the black style.
- In Git Bash on Windows, very long heredocs with Python code fail to parse;
  write files with an editor/tool, use heredocs only for short snippets.
- OpenCV CSRT: `cv2.TrackerCSRT_create` exists in opencv-contrib 4.12; the
  box width drifts smaller over long runs (60 -> 43 px over 10 s on the
  synthetic clip). The 10 % pad hides it; consider periodic re-detection.

## Performance reference (i7-14700T laptop, CPU only)

| Task | Speed |
|---|---|
| CSRT tracking @640 px | ~42-56 fps |
| YOLOX-s @640 + YuNet, per detector pass | ~0.15-0.2 s |
| faster-whisper small int8 | ~3.5x real time |
| Export 640x360 (libx264 veryfast) | ~350 fps |

## Log

- 2026-10-06: Project created. Core, vision, audio, UI, tests, smoke scripts,
  docs, installer scaffolding. Reference repo research in docs/RESEARCH.md.
