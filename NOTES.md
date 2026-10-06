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

## First real-footage feedback and fixes (2026-10-06 pm, v0.1.3)

Wyatt tested v0.1.2 on body-cam footage (samples/, two Axon clips 720p30, 14 and
47 min). Findings and what changed:

- **Tracking unusably slow.** CSRT cost 72 ms/frame on real 720p (vs 20 on the
  synthetic clip), the backward pass doubled it, and a resident ONNX Runtime
  session made everything 2.5-5x slower still because ORT worker threads
  spin-wait between inferences. Fixes in vision/tracker.py + detector.py:
  KCF instead of CSRT; adaptive work-frame size so the object is ~110 px;
  process every 2nd frame (interpolate); re-create the tracker only when a
  correction moves the box (IoU < 0.85); ORT `intra_op.allow_spinning=0` and
  4 intra-op threads; sparser search while lost; backward pass stops after
  4 s lost. Result: 13 -> ~125 source fps with detector, ~200 without
  (14-min clip ≈ 2-3 min, used to be ~30). Tracking now runs as a background
  job and the box fills in live on the timeline; the UI stays usable.
- **Transcription "did nothing".** Log showed Silero VAD removed 12 of 14
  minutes as non-speech (traffic noise, quiet voices). VAD is now off by
  default (setting), whisper's no-speech/filler filter drops hallucinations
  ("you", "Thank you."), and segments stream into the transcript view as they
  are recognised. Measured on the traffic stop: VAD on 137 words, off 199.
- **Auto-detect found nothing.** Detectors work on the crash-scene clip (YuNet
  @1280: 38 faces in 80 sampled frames; YOLOX-tiny: screens, phones). Most
  likely the installed copy had no models (download blocked or cancelled).
  The installer now **bundles all detection models + whisper small** (~550 MB
  installer); `-NoModels` for a slim build. Face conf default 0.5, YuNet at
  full 1280 px, YOLOX-tiny default, stride 5. CenterFace is wired as an
  option but was slower (233 ms) and found fewer faces here; keep YuNet.
- **"No selection on audio"**: drag on the timeline's Audio lane now creates
  an audio redaction directly (no transcript needed).
- Jobs dock (ui/jobs.py) replaces the modal progress dialog for transcribe,
  track, detect, export and hashing; `partial()` streams results to the UI.
- `--selftest --transcribe <media>` also checks speech recognition.

Benchmarks live in NOTES "Performance reference"; scratch scripts were in the
session scratchpad (bench_*.py) - recreate from the numbers if needed.

## Build, versioning, releases, auto-update (added 2026-10-06)

- `installer\build.ps1` = the release button. It bumps the patch version
  (`scripts\bump_version.py`, single source of truth `videoredact/__init__.py`,
  mirrored to pyproject.toml), builds with PyInstaller (one-folder, spec in
  installer/), compiles the Inno Setup installer into `Published\`, writes
  `latest.json` + `SHA256SUMS.txt`, commits "Release X.Y.Z", tags `vX.Y.Z`,
  pushes. `-Publish` (or `scripts\publish_release.ps1`) creates the GitHub
  release and uploads the installer. The publish script uses `gh` if present,
  else the token Git Credential Manager already holds for github.com.
- `Published\` and `bin\` are git-ignored (installer ~hundreds of MB; GitHub
  file limit is 100 MB). Releases are the distribution channel.
- Auto-update lives in `videoredact/updater.py` + `MainWindow.check_for_updates`.
  It reads `releases/latest`, looks for an asset named exactly
  `VideoRedact-X.Y.Z-Setup.exe`, downloads to %TEMP%, runs it with
  `/SILENT /CLOSEAPPLICATIONS /RESTARTAPPLICATIONS`, and quits. Inno `AppId`
  must never change or upgrades become side-by-side installs.
- Installer is `PrivilegesRequired=lowest` so standard users can install and
  auto-update per-user (`%LocalAppData%\Programs\VideoRedact`); admins get the
  all-users choice in the dialog.
- Frozen build: data files live in `_internal\` (`sys._MEIPASS`); `paths.resource_root()`
  and `media.ffmpeg_exe()` look there. stdout/stderr go to
  `%LocalAppData%\VideoRedact\logs\videoredact.log` (Help ▸ Open log folder);
  uncaught exceptions show a dialog and are logged.
- FFmpeg shipped in the installer: `bin\ffmpeg.exe`. Test builds copy the
  imageio-ffmpeg binary (a GPL build). Before wide distribution drop an LGPL
  build there (BtbN "lgpl" variants) so the whole package stays permissive.
- Inno Setup on this laptop installed per-user via winget:
  `%LocalAppData%\Programs\Inno Setup 6\ISCC.exe` (build.ps1 checks that path too).

## Gotchas

- Store Python virtualizes `%LOCALAPPDATA%`: anything the venv writes to
  `C:\Users\<you>\AppData\Local\VideoRedact` really lands in
  `...\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_*\LocalCache\Local\VideoRedact`.
  The frozen EXE (a normal process) sees the real folder, so models downloaded
  while developing are invisible to the installed app and vice versa. Copy them
  across or use a python.org interpreter for the venv. HF symlink warnings are
  harmless (HF_HUB_DISABLE_SYMLINKS_WARNING=1 is set in app.py).
- `VideoRedact.exe --selftest [--quiet] [media]` verifies an installation
  (ffmpeg, libs, models, detectors, export) and writes the report to the log.
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

| Task (real 720p30 body-cam unless noted) | Speed |
|---|---|
| cv2 decode 720p / grab only | 777 / 2550 fps |
| KCF update @512 (small box) | 3.4 ms; CSRT 72 ms; MOSSE 0.2 ms |
| Tracker end-to-end, step 2, with YOLOX-tiny snaps | ~125 source fps (200 without detector) |
| YuNet @1280 / @960 | 29 / 16 ms per frame |
| YOLOX-tiny / YOLOX-s | 26 / 79 ms (with ORT spinning off) |
| CenterFace @1280 | 233 ms (not default) |
| faster-whisper small int8, VAD off | ~7x real time |
| Export 640x360 (libx264 veryfast) | ~350 fps |

## Log

- 2026-10-06 (pm): Release pipeline done. v0.1.1 and v0.1.2 built, installed
  per-user, self-tested, published to GitHub Releases. Verified the real update
  path: installed 0.1.1 -> fetch latest -> download 0.1.2 -> silent upgrade ->
  installed app reports 0.1.2. Current installed copy on this laptop: 0.1.2
  at %LocalAppData%\Programs\VideoRedact. Next release: `.\installer\build.ps1 -Publish`.
- 2026-10-06: Project created. Core, vision, audio, UI, tests, smoke scripts,
  docs, installer scaffolding. Reference repo research in docs/RESEARCH.md.
