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
  people (YOLOX) every few frames, linked into tracks and *carried* by a
  tracker between detections so turned or blurred faces stay covered. Runs on
  several CPU cores in parallel.
- Draw a box around anything and the object is tracked forward and backward
  (ViTTrack, scale-adaptive), and re-acquired when it leaves the frame and
  comes back later.
- Export verification: faces are re-detected in the redacted output and any
  left visible are listed with timestamps before you release the file.
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

The installer bundles every model (face/object detectors and the `small`
speech model), so installed copies work with no internet. When running from
source, models are downloaded to `%LOCALAPPDATA%\VideoRedact\models` on first
use (`python scripts\fetch_models.py --whisper small` pre-fetches them).

Long operations (transcribe, track, detect, export) run as background jobs in
the Jobs panel; the window stays usable and results stream in live. Tracking a
box through a 14-minute 720p clip takes about 2-3 minutes on an 8-core CPU.

FFmpeg: the dev environment uses the binary from the `imageio-ffmpeg` wheel.
The installer ships an LGPL build in `bin\ffmpeg.exe`. Override with
`VIDEOREDACT_FFMPEG=<path>`.

## Building the installer and publishing a release

```powershell
winget install JRSoftware.InnoSetup            # once per machine
C:\dev\venv-videoredact\Scripts\python -m pip install pyinstaller
.\installer\build.ps1                           # bump patch version, build, write Published\VideoRedact-<ver>-Setup.exe,
                                                # commit "Release <ver>", tag v<ver>, push
.\installer\build.ps1 -Publish                  # ...and also create the GitHub release with the installer attached
.\scripts\publish_release.ps1                   # publish the last build separately
```

Every build gets a new version number (patch increment by default; `-Bump minor`
or `-Bump major`, `-NoBump` to rebuild the current version). Output goes to
`Published\` (git-ignored because installers are hundreds of MB) together with
`latest.json` and `SHA256SUMS.txt`.

**Auto-update:** installed copies check the latest GitHub release a few seconds
after startup (and from Help ▸ Check for updates). If a newer
`VideoRedact-X.Y.Z-Setup.exe` asset exists, the user is offered *Install now*;
the installer is downloaded to %TEMP%, run silently, closes and reopens the app.
Only the public release metadata is requested from GitHub; media never leaves
the machine. The check can be turned off in Settings ▸ Updates.

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
