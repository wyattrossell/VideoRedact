# VideoRedact

Free, offline video and audio redaction for law enforcement. Runs entirely on an
ordinary Windows PC (no GPU required). MIT licensed.

**Audio**
- Transcribes speech on the CPU (faster-whisper) with word-level timestamps.
- Select words, phrases or sentences in the transcript and redact them.
- Redact every occurrence of a word or phrase with one click.
- Rule-based suggestions for sensitive speech: phone numbers, SSNs, names after
  cues such as "my name is", street addresses, dates of birth, license plates.
- Styles per redaction: beep (configurable frequency), silence, low tone, noise.

**Video**
- Automatic detection of faces (YuNet), screens / laptops, phones, documents,
  people (YOLOX, COCO classes) with temporal linking into tracks.
- Draw a box around anything and the object is tracked forward and backward,
  and re-acquired when it leaves the frame and comes back later.
- Fix a track by dragging its box; the correction becomes a keyframe.
- Styles per region: solid black, blur, pixelate; rectangle or ellipse.

**Evidence handling**
- Preview is a faithful rendering of the export (same code path).
- Export writes the redacted media plus a PDF and CSV **redaction report**
  (every redaction, who/when, SHA-256 of source and output, full audit log)
  and a reopenable `.vrproj` project file. The original is never modified.

## Status

Early but working end-to-end (October 2026). See [NOTES.md](NOTES.md) for the
developer log and [docs/ROADMAP.md](docs/ROADMAP.md) for what is next.

## Running from source (Windows)

```powershell
# 1. Python 3.11+ from python.org (the Microsoft Store build hits path-length limits)
# 2. Create the environment at a SHORT local path and install:
.\scripts\setup_dev.ps1          # creates C:\dev\venv-videoredact and installs requirements
# 3. Run:
.\scripts\run_dev.ps1            # or: C:\dev\venv-videoredact\Scripts\python -m videoredact
```

Detection models (~75 MB) and the speech model (~500 MB for `small`) are
downloaded to `%LOCALAPPDATA%\VideoRedact\models` on first use. For machines
without internet run `python scripts\fetch_models.py --whisper small` on a
connected machine and copy that folder.

FFmpeg: the dev environment uses the binary from the `imageio-ffmpeg` wheel.
The installer ships an LGPL build in `bin\ffmpeg.exe`. Override with
`VIDEOREDACT_FFMPEG=<path>`.

## Tests

```powershell
C:\dev\venv-videoredact\Scripts\python -m pytest            # unit tests (fast)
C:\dev\venv-videoredact\Scripts\python scripts\make_sample.py   # synthetic test clip with TTS speech
C:\dev\venv-videoredact\Scripts\python scripts\smoke_video.py   # tracking + export end to end
C:\dev\venv-videoredact\Scripts\python scripts\smoke_transcribe.py small
C:\dev\venv-videoredact\Scripts\python scripts\smoke_ui.py       # offscreen UI
C:\dev\venv-videoredact\Scripts\python scripts\smoke_playback.py # real window playback
```

## Components and licenses

| Component | Use | License |
|---|---|---|
| PySide6 / Qt 6 | UI, audio output | LGPL-3 |
| OpenCV (contrib) | decoding, CSRT tracker, YuNet runtime | Apache-2.0 |
| ONNX Runtime | YOLOX inference | MIT |
| faster-whisper / CTranslate2 | speech-to-text | MIT |
| FFmpeg | decode / encode | LGPL build |
| YuNet face detector | faces | MIT |
| CenterFace (optional) | faces, higher recall | MIT |
| YOLOX-s / tiny | screens, documents, phones, people | Apache-2.0 |
| reportlab | PDF report | BSD |

No AGPL/GPL code is used (this rules out Ultralytics YOLO and boxmot).

## Project layout

```
videoredact/
  app.py              entry point
  paths.py            model/settings directories, persisted settings
  core/model.py       data model: transcript, redactions, tracks, project (.vrproj JSON)
  core/media.py       ffmpeg discovery, probe, audio decode, FrameReader
  core/audio_redact.py  beep/silence/tone/noise rendering
  core/video_redact.py  black/blur/pixelate rendering
  core/export.py      export pipeline (OpenCV -> FFmpeg pipe), sidecars
  core/report.py      CSV + PDF report
  audio/transcribe.py faster-whisper wrapper, word/phrase matching
  audio/pii.py        rule-based PII suggestions
  vision/models.py    model registry + downloader
  vision/detector.py  YuNet + YOLOX detectors
  vision/tracker.py   single-object CSRT tracking with loss detection and re-acquisition
  vision/auto_detect.py  detect-every-N-frames + IoU association into tracks
  ui/                 PySide6 UI (main_window, player, video_view, transcript_view, timeline, panels, dialogs)
scripts/              dev helpers, sample generator, smoke tests, model fetcher
installer/            PyInstaller spec + Inno Setup script + build script
tests/                pytest unit tests (+ tests/data sample media)
docs/                 research notes, roadmap
```
