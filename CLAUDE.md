# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is
VideoRedact: offline, CPU-only video/audio redaction desktop app (Python 3.11+,
PySide6, OpenCV, ONNX Runtime, faster-whisper, FFmpeg) for law enforcement,
distributed free as a Windows installer. MIT licensed; **never add GPL/AGPL
dependencies or copy GPL/AGPL code** (no Ultralytics, boxmot, StrongSORT).

## Start here
1. Read `NOTES.md` (state, decisions, gotchas, machine setup).
2. Python env lives at `C:\dev\venv-videoredact` (create with
   `scripts\setup_dev.ps1`). Run everything with that interpreter:
   `C:\dev\venv-videoredact\Scripts\python.exe`.
3. Fast checks before/after changes:
   - `python -m pytest` (unit tests, <2 s)
   - `python scripts\smoke_video.py` (tracking + export, ~30 s)
   - `python scripts\smoke_ui.py` (offscreen UI, ~5 s)
   - `python scripts\smoke_playback.py` (real window, ~5 s)
   If `tests/data/sample.mp4` is missing run `python scripts\make_sample.py`.

## Conventions
- Times in seconds (float); frame indices int; boxes normalized `BBox(x,y,w,h)`.
- Any user action that changes redactions must go through `Project` helpers
  (`add_audio_redaction`, `add_video_track`, ...) or call `project.log()` so
  the audit log in the report stays complete.
- Long work (transcribe, detect, track, export) runs via `ui/workers.run_task`
  with `progress(p, msg)` and `cancel()` callbacks; never block the GUI thread.
- Preview must render through the same functions as export
  (`core/video_redact.redact_frame`, `core/audio_redact.apply_redactions`).
- Keep CPU/memory modest: detectors on <=960 px frames, tracker on <=640 px,
  stream audio instead of loading whole files when possible.
- Models: register in `vision/models.py` with license + min size; fetch via
  `ensure_model`. Only permissive licenses.
- On Windows + Git Bash, do not write large Python files with shell heredocs
  (parser breaks); use the editor tools.

## Repo map
See README.md "Project layout". UI entry: `videoredact/app.py` ->
`ui/main_window.py`. Preview player: `ui/player.py` (own implementation; do
not switch back to QMediaPlayer, see NOTES.md).

## When finishing a session
Update `NOTES.md` (Current state / Log / Gotchas) and commit + push so the
next machine picks up where this one left off.
