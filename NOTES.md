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
  Black box is the default. Since v0.1.10 pixelate has a 16 px block floor and
  "blur" is a heavy blur (area-average to ~6 samples); light Gaussian blur is gone
  because it is partially reversible (docs/CJIS_COMPLIANCE.md section 5).
- **Offline by default** (v0.1.10): `settings.offline=True`; `check_updates=True` since
  v0.1.11 but notify-only (GitHub release metadata, link, never installs);
  `videoredact/offline.py` runs on package import and blocks `openvino_telemetry`
  (it posts to Google Analytics when no consent file exists). Every new dependency
  must pass `python -m videoredact.app --selftest --offline-check tests/data/sample.mp4`
  (exit 0, "0 blocked connection attempt(s)") before it ships.
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

## Round 3: speed + detection/tracking effectiveness (2026-10-06 night, v0.1.5)

Wyatt asked to focus on performance and detection/tracking quality. Measured on
samples/ (720p30 Axon clips), i7-14700T:

- **Tracker: ViTTrack (opencv_zoo, Apache-2.0) is now default.** 5 ms/frame
  regardless of box size (KCF was 15 ms on the laptop-screen box, CSRT 45),
  and it adapts scale (KCF froze at the initial size). Models registered as
  `vittrack`; KCF/CSRT remain selectable in Settings. NanoTrack does not load
  in OpenCV 5's new ONNX importer; skipped. Note: OpenCV 5 exposes
  `getTrackingScore()` on every tracker and KCF returns 0 -> only use the
  score for ViT.
- **Auto-detect rewrite** (vision/auto_detect.py): detections every N frames
  with two-pass IoU association; tracks that miss a detection are *carried*
  with ViTTrack from the exact last-detection frame (ring buffer) until
  re-detected / lost / 3 s; confident-or-not single detections are kept
  (recall first, flagged "single detection - review"); fragments within 1 s
  and nearby are linked; **parallel chunk workers** (spawn, 4 on this box,
  2-3 on an 8-core) merged by IoU at chunk boundaries. Detectors now get the
  **full-resolution frame** (the 960 px copy lost ~7 % of small faces).
  3-min crash-scene slice, faces: old pipeline 82 % of reference detections
  covered @173 fps -> new 100 % @141 fps (parallel), fragments 127 -> 282
  (because singles are kept; linking already merged adjacent ones).
- **Post-export verification** (core/verify.py, Export dialog checkbox, on
  by default): re-detect faces in the *output* every 15 frames, report any
  detection < 60 % covered with timestamps; CSV sidecar; "Jump to first".
  Borrowed from redactcam's coverage check.
- Hardware encoders (QSV/NVENC/MF) are **not** faster than libx264 at 720p
  (all ~400-500 fps) - export is decode/python bound; not pursued.
- multiprocessing in the frozen build needs `multiprocessing.freeze_support()`
  first thing in main() (done). `--selftest --detect <media>` exercises it.
- Research reports saved: docs/RESEARCH-commercial.md, RESEARCH-legal.md,
  RESEARCH-technical.md. Roadmap re-prioritised from them.

## Round 2 feedback (2026-10-06 evening, v0.1.4)

- **"Warnings when changing options" + "video redaction not working"** were one
  bug: Qt's `QComboBox.itemData()` returns our `str`-based enums as plain `str`,
  so every style change hit `'str' object has no attribute 'value'`, and the
  same crash fired inside `_run_tracking` right after the placeholder track was
  added, before the tracking job started -> the box existed on one frame only.
  Fix: `model.as_enum()` coercion at every UI entry point and in the
  serializer/renderers; combos now store plain values. Regression test:
  tests/test_ui_styles.py (drives the real combos offscreen).
- Lesson: anything that goes through a QVariant comes back as a base type.
  Never store enum members as item data; store `.value` and coerce on read.
- scripts/smoke_realvideo.py drives the real window on samples/ end to end
  (track -> style change -> export -> pixel check). Run it before releasing.

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

- Inno Setup's `CloseApplications` (Restart Manager) hung indefinitely on this
  laptop when upgrading 0.1.5 -> 0.1.6 with no VideoRedact process running;
  `/NOCLOSEAPPLICATIONS` installs fine in ~40 s. The in-app updater now passes
  it and relaunches the app via `cmd /c setup && start exe`. For manual silent
  installs use PowerShell `Start-Process` (Git Bash mangles `/PID`, `/LOG=`
  style flags into paths; `taskkill /PID` silently does nothing there).

- Training on Windows (train/): YOLOX 0.3's own trainer ran at 5 s/iteration
  with the GPU idle (the loader and GPU step each measured < 0.2 s), so
  train/train_loop.py is a minimal loop around exp.get_model(); YOLOX's
  sampler is infinite, so epochs must be fixed-length over a persistent
  iterator; its fast COCO eval needs a C++ build, so pycocotools is
  monkeypatched in. `pip install yolox` fails building old onnx: install the
  deps by hand and yolox with --no-deps. venv at C:\dev\venv-train (torch cu128).
- Never write O(n^2)-per-step Python clustering for thousands of embeddings:
  the first bench/face_identities.py spent 3.4 CPU-hours on 1,624 vectors
  before being killed; scipy's linkage does it in under a second.
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
| ViTTrack / KCF / CSRT update (large box @512) | 5 / 15 / 45 ms |
| Auto-detect faces, stride 5, carry, 4 workers | ~140 source fps (single process ~85) |
| Post-export face verification, every 15th frame | ~2x real time |
| Hardware H.264 encoders vs libx264 @720p | no gain (400-500 fps all) |
| YuNet @1280 / @960 | 29 / 16 ms per frame |
| YOLOX-tiny / YOLOX-s | 26 / 79 ms (with ORT spinning off) |
| CenterFace @1280 | 233 ms (not default) |
| faster-whisper small int8, VAD off | ~7x real time |
| Export 640x360 (libx264 veryfast) | ~350 fps |

## Log

- 2026-10-28 (evening): video redaction rework after Wyatt's feedback (auto-detect
  over-redacts, objects become faces/persons; manual boxes must cover before and
  after; hotkeys; audio untouched). Measured on the reference clip with contact
  sheets per confidence bin: only face tracks with 3+ detections at avg conf >= 0.7
  are reliably real; head-fallback ~1/3 real; single detections ~45 %. New default
  "precise" profile (AutoDetectOptions / confirm_track): 2,313 -> 362 tracks, 126.5
  vs 92.7 fps; recall profile kept as an option. Manual tracker: hold 12 frames
  after loss, bridge gaps <= 2 s, stop 8 s after the object is gone (notepad: 17
  fragments / 13 s -> 1 span / 22.7 s of the human's 23 s; no more 85 s searches
  with false re-acquisitions). Drag = track immediately, Shift+drag = dialog.
  Keys N/Shift+N, Ctrl+Z undo delete, E, X split, T. Settings rev 2 resets
  detect_head. All in bench/results/PRECISION_PROFILE.md. v0.1.15 built,
  installed per-user, self-test ALL OK / 0 network attempts; not yet published
  (0.1.14 is not published either). Installed copy on this laptop: 0.1.15. Next: Wyatt's own
  hand-redacted project files as ground truth for faces (see chat 2026-10-28).
- 2026-10-28 (pm): v0.1.13 built (BBox float fix + inert emit_person_boxes /
  doc_model hooks) but NOT published: its Tools > "Check detection models" fails
  because the optional yolox_tiny_doc entry has no download source. v0.1.14
  fixes that (required_models() skips absent optional models; regression test),
  built, installed per-user, self-test ALL OK with 0 network attempts. Release
  commit + tag pushed; the GitHub release (publish_release.ps1) is still to be
  run by Wyatt. Installed copy on this laptop: 0.1.14. Gotcha: a silent install
  exits 5 ("file in use") while the app is open; check Get-Process VideoRedact
  first, worker processes carry --multiprocessing-fork.
- 2026-10-28: Reference-pair analysis finished (bench/results/REFERENCE_TANYARD.md).
  The hand-redacted copy covers one notepad for 23 s and nothing else; we cover
  20/21 of those boxes only through accidental whole-body person boxes and add
  17k extra box-frames. Tried to close the document gap with a YOLOX-tiny
  fine-tune on OWLv2 pseudo-labels (train/, C:\dev\venv-train, RTX 3050 Ti):
  three runs, the honest one (reference clip held out) fits its own labels
  (document AP50 0.56) and scores 0.00 on the held-out clip; the teacher put
  zero document boxes inside the notepad span and brightening frames does not
  help it. Not shipped; the doc_model hook stays inert. Needs hand-labelled
  paper from many scenes. Identity grouping of face tracks (SFace, scipy
  clustering, bench/face_identities.py) measured as the basis for
  "everyone except X" and long-gap track linking; numbers in the report.
  Open decision for Wyatt: emit_person_boxes default (see report).
- 2026-10-16: Readiness review before user testing (v0.1.12). Ran everything:
  65 tests, six smokes, installed build on a real clip with --detect --transcribe
  --offline-check. Findings fixed: (1) transcription language auto-detect produced
  non-Latin garbage and ran 4x slower on noisy body-cam openings -> default
  language "en" + one-time settings migration (settings_rev); (2) an independent
  code review found export/batch silently overwriting files, arrow keys dead after
  clicking the video/timeline/transcript (widgets swallowed them; now an
  app-level event filter), transcript wiped before a failed model load, export
  allowed mid-tracking, delete with no confirm, raw HTML in the update dialog ->
  all fixed (see CHANGELOG 0.1.12). Deferred: audio-only waveform decode on the
  UI thread, possible QThread warning on quit mid-detection, NVIDIA tier, LGPL
  FFmpeg swap. docs/USER_TESTING.md is the tester guide. smoke_playback shows
  PLAYBACK PROBLEM only when the CPU is saturated by other smokes; run it alone.
  Gotcha: smoke_ui must call _delete(..., confirm=False) (new confirmation dialog).

- 2026-10-15 (later): Phase 5 (updates and release) + licensing audit. Updater is
  notify-only (background check, 5 s timeout, silent offline, link to the release
  page; download/launch code removed and a test forbids re-adding it); startup
  check default ON again (user's Phase 5 ask; public metadata only; Settings >
  Network turns it off). VERSION + CHANGELOG.md + `bump_version.py --check/--notes`
  (build.ps1 promotes [Unreleased], publish_release.ps1 uses the section as notes).
  .github/workflows/ci.yml: pytest + offline self-test on push/PR; benchmark on the
  synthetic tests/data/sample.mp4 on each published release, results attached to
  the release. docs/LICENSES.md (FFmpeg GPL build still to swap; Ultralytics AGPL,
  InsightFace weights non-commercial, SAM licence summarised) + tests/test_licenses.py.
  Benchmark rerun after Phase 4 (bench/results/phase4.json, COMPARISON.md): 83.8 fps
  auto-detect vs 59.8 baseline, export 198 vs 202; the first run exposed a float32
  JSON crash in the manifest writer (fixed). tests/test_tracks.py covers back-fill,
  linking, mask dilation, audit log. Released as v0.1.11.

- 2026-10-15: Phase 4 (evidence integrity). core/audit.py writes a self-hashed
  `<output>_audit.json` per export (operator, versions, model SHA-256s, thresholds,
  all tracks/boxes/edits, verification); export re-hashes the source and flags a
  change; PDF/CSV print the manifest hash and a model table. Pixelate floor 16 px,
  blur made heavy, black stays default. Offline hardening in videoredact/offline.py;
  `--selftest --offline-check` blocks sockets: first run caught openvino_telemetry
  -> www.google-analytics.com (fixed by excluding/blocking the package). Settings
  offline=True, check_updates=False. docs/CJIS_COMPLIANCE.md + PHASE4_NOTES.md.
  tests/test_audit.py (14 tests). Released as v0.1.10; installed on this laptop,
  frozen `--selftest --offline-check` = 0 connection attempts, manifest verifies.
  Gotcha: do not run `installeruild.ps1` with `2>&1` from Windows PowerShell 5.1;
  PyInstaller's stderr becomes a NativeCommandError and the script aborts after
  bumping the version (revert pyproject.toml + __init__.py, then rerun). Run it
  plainly or via `powershell.exe -File installeruild.ps1 -Publish > log 2>&1`.

- 2026-10-15 (late): v0.1.9 built (700 MB incl. openvino), published, installed
  per-user, self-test OK on the frozen build: tier=intel via openvino:GPU.0,
  YuNet + YOLOX on the iGPU, parallel auto-detect of the first 1800 frames of
  the crash-scene clip in 20.6 s with the same 159 tracks (0.1.8: 40.4 s).
  Installed copy on this laptop: 0.1.9.

- 2026-10-15 (Phase 3, speed): measured decode paths (OpenCV fastest at full
  res; FFmpeg proxy pipes ~500 fps with QuickSync for >1280 px sources),
  inference engines (OpenVINO on the Intel UHD 770 iGPU: YOLOX-tiny 11 ms,
  YuNet 8 ms; ORT CUDA/TensorRT and the ORT OpenVINO EP could not be loaded
  from pip on this machine). Built: core/streams.py (decode/encode threads),
  vision/runner.py (OpenVINO API runner with disk cache + ORT fallback), YuNet
  on ORT/OpenVINO with raw NHWC u8 input (11 ms end to end vs 30 via OpenCV
  DNN, identical boxes), tracker pool (ViT create was 20 ms x 249/clip),
  cv2.copyTo mask compositing, scene-cut + adaptive stride (coverage gate
  100 %), cheaper carrying, network-share staging to local disk (+40 % export),
  batch queue (File > Batch), segment-parallel export (opt-in; no gain here:
  two concurrent exports = 1.2x). Results table in docs/PHASE3_NOTES.md.
  Gotcha: `trackers` pulls plain opencv-python (shadows contrib);
  `openvino` must be installed in the app venv for the Intel tier.

- 2026-10-14 (late): v0.1.8 built (bundle now staged on local disk: PyInstaller's
  COLLECT failed creating dirs on the UNC share; installer/build.ps1 passes
  /DDistDir to Inno), published, installed per-user, self-test OK: frozen
  parallel auto-detect with the Phase 2 pipeline = 159 face tracks / 1800
  frames in 40 s (was 79 tracks / 9 s with the v0.1.7 pipeline - the new
  default redacts far more but costs ~4x). Installed copy on this laptop: 0.1.8.
  Gotcha: `pip install trackers` drags in opencv-python which shadows
  opencv-contrib (KCF/CSRT vanish) - setup_dev.ps1 reinstalls contrib last.

- 2026-10-14 (Phase 2): pluggable detectors (vision/detectors/), face
  evaluation on our frames (YuNet@1280 + multi-scale x1.5 wins: 96 % of
  consensus faces, SCRFD-10G 69 %, CenterFace 29 %; EgoBlur needs a manual
  click-through download), head fallback (person box -> head region, merged
  into the face lane), 25 % face padding + ellipse/rounded masks, BoT-SORT
  association via Roboflow trackers (Apache) made purely additive (raw boxes
  redacted; tracker only supplies identity), lead-in/persistence 15 frames,
  review flags + F/Shift+F + [ ] = keys, train/ fine-tune scaffold, Grounding
  DINO adapter (9 s/frame on CPU -> not viable). Measured on the 3-min
  crash-scene slice: coverage stays 100 %, redacted frame-boxes 15k -> 69k
  (turned heads, distant faces, persistence), 57 fps parallel (was 172),
  tracks 282 -> 464 (224 singletons flagged "single"). Details:
  docs/PHASE2_NOTES.md. SCRFD adapter had an x/y anchor swap at first - fixed.

- 2026-10-14: Phase 1 benchmark harness in bench/ (make_clips, extract_frames,
  preannotate -> Label Studio local, ls_to_yolo, run_bench). Baseline on 11 x 45 s
  clips: auto-detect 60 fps single-process (86 % of time in detection, 72 ms per
  detect call with face+screen+document+phone), export 202 fps (66 % encode+pipe,
  29 % decode, 5 % redact). Clips are < 1500 frames so the parallel path did not
  engage (parallel column == single). Accuracy pending Wyatt's labels.
  Label Studio 1.23 installed at C:\dev\venv-labelstudio (bench\label.ps1).
  docs/UPGRADE_PLAN.md written; waiting for approval before changing the pipeline.

- 2026-10-14: Phase 0 audit. docs/ARCHITECTURE_CURRENT.md documents the
  pipeline and 24 weaknesses with file:line refs. New vision/accel.py detects
  hardware (dev laptop: RTX 3050 Ti + UHD 770, only CPU runtimes installed ->
  tier=cpu) and feeds a provider chain (TensorRT/CUDA -> OpenVINO -> DirectML
  -> CPU) into detector._ort_session with CPU fallback; `--selftest` prints
  the tier and why a higher one was not used. Accelerated wheels are not
  bundled yet (Phase 1).

- 2026-10-06 (night): v0.1.5 published and installed; frozen self-test ran
  parallel auto-detect (2 workers, 79 face tracks / 1800 frames in 13 s) and a
  full 47-min export (509 s, ~166 fps). v0.1.6 adds **clip-range export**
  (Export dialog "Export only a time range"; verification maps back to source
  time; report notes the range) - a 60 s clip exports in 12 s - and the
  self-test exports only 20 s. Installed copy on this laptop: 0.1.7 (0.1.7 = updater skips Restart Manager).

- 2026-10-06 (evening): v0.1.4 built, published, installed per-user, self-test
  OK. Installed copy on this laptop: 0.1.4. Fixes the str-enum crash behind
  "warnings when changing options" and tracking not starting.

- 2026-10-06 (late pm): v0.1.3 built (632 MB, models bundled), published,
  installed per-user and self-tested offline incl. speech recognition on the
  real traffic-stop clip (48 words in first 90 s, 29 s). Installed copy on this
  laptop: 0.1.3. Export of the full 14-min 720p clip in the frozen build took
  166 s (~150 fps).

- 2026-10-06 (pm): Release pipeline done. v0.1.1 and v0.1.2 built, installed
  per-user, self-tested, published to GitHub Releases. Verified the real update
  path: installed 0.1.1 -> fetch latest -> download 0.1.2 -> silent upgrade ->
  installed app reports 0.1.2. Current installed copy on this laptop: 0.1.2
  at %LocalAppData%\Programs\VideoRedact. Next release: `.\installer\build.ps1 -Publish`.
- 2026-10-06: Project created. Core, vision, audio, UI, tests, smoke scripts,
  docs, installer scaffolding. Reference repo research in docs/RESEARCH.md.
