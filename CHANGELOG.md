# Changelog

All notable changes to VideoRedact. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/). The section for a version is used
verbatim as its GitHub release notes (`scripts/bump_version.py --notes`). Write entries under
**Unreleased** as you go; `installer/build.ps1` promotes them when it bumps the version.

## [Unreleased]

### Added
- Select a face track and press P (Video menu: "Find this person in every face track") to find
  every other face track of the same person with the bundled SFace identity model. The matches
  are noted in the Redactions panel and can be disabled or deleted in one go, or you can keep
  only that person redacted and disable every other face track. Runs locally; nothing is stored.

## [0.1.17] - 2026-10-31

### Changed (tracking rebuild, Phase 1 and 3)
- A box you draw around a face (or a phone, screen or document) is now tracked by a new engine
  that works at full resolution, checks the face detector around the box every few frames and
  right away whenever the tracker is unsure, keeps a bank of recent views of the object, and
  uses a face identity model (SFace, Apache-2.0, bundled) to make sure a face it finds again
  after a loss is the same person. On 24 confident faces from our clips one drawn box now
  covers 97 % of the face's visible frames (was 86 %); phones 87 % (unchanged) with a third of
  the overrun. Numbers and method: `bench/results/TRACKING_V2.md`, `bench/track_eval.py`.
  Objects without a detector (plates until the next release, tattoos, notepads) keep the
  previous engine, which measured better on them. Settings: `track_engine` = anchored (default)
  | anchored-all | classic.
- Draw a box in a gap of the selected track (within 10 s of where it stopped) and it continues
  that track instead of starting a new one; press J first to force that from anywhere. Ctrl+Z
  undoes a join.
- The tracker's face detector runs on the CPU ONNX runtime (1-2 ms per region) and reports
  landmarks; the Intel GPU path recompiled for every box size.
- ViTTrack keeps reporting a box down to an internal score of 0.05 (was 0.20); the engines apply
  their own hysteresis.

## [0.1.16] - 2026-10-29

### Fixed
- Auto-detect on sources wider than 1280 px (every 1080p body-cam file) could run on a frozen
  picture: the detection pipe tried Intel QuickSync first and on some Intel machines that
  decoder returned the first frame over and over, so every auto-detected box stayed where the
  face was on frame one while the exported video looked normal. The pipe now uses DirectX
  (d3d11va) decode with a software scale, every hardware decoder is checked on its first 12
  frames and rejected if they are identical, and `--selftest <file>` reports "proxy decode"
  with the decoder used. 720p sources were never affected. If you auto-detected a 1080p file
  with an earlier version, run Auto-detect again.

## [0.1.15] - 2026-10-28

### Changed (video redaction rework: predictable, high-confidence, faster to review)
- Auto-detect keeps only confident detections by default: a face must be detected 3+ times with
  average confidence 0.7+, boxes taller than 60 % of the frame are rejected, whole-body "person"
  boxes and the head-fallback heuristic are off. Phones/screens/documents need 2+ detections at
  0.4+. On the 45-min reference clip this is 362 tracks instead of 2,313 (every kept face bin
  was checked by eye; the dropped bins were mostly phones, lights, hands and car parts) and
  detection runs 36 % faster. Untick "Only keep confident detections" in the Auto-detect dialog
  for the old find-everything behaviour. Numbers: `bench/results/PRECISION_PROFILE.md`.
- A manually drawn box is tracked forward and backward immediately with the last-used label,
  shape and style (Shift+drag opens the dialog; Video menu toggles this). The tracker now holds
  the box briefly after losing the object and bridges gaps of up to 2 s between re-acquisitions,
  so a manual redaction no longer flickers: the reference notepad went from 17 fragments
  covering 13 of 22 s to one span covering 22.7 s (the hand-redacted copy covers 23 s). It stops
  once the object has been out of view for 8 s instead of searching to the end of the video
  (which took 85 s and produced a false re-acquisition four minutes later); "keep searching to
  the end" remains an option in the dialog.
- Review keys: N / Shift+N next/previous redaction, Delete now undoable with Ctrl+Z (no
  confirmation dialog), E enable/disable, X split track at the playhead, T track the selected
  box from the playhead. Quick guide (F1) lists them all.
- Settings migration (rev 2): the saved "head fallback" choice is reset to off once.
- Audio redaction is untouched.

## [0.1.14] - 2026-10-28

### Fixed
- Tools -> "Check detection models..." failed on every install with "Model 'yolox_tiny_doc.onnx'
  is optional and has no download source" (regression in 0.1.13: the optional fine-tuned
  detector was registered but never ships). Optional models are now skipped and listed as
  "optional, not installed"; the self-test labels them the same way.

## [0.1.13] - 2026-10-28

### Added
- Auto-detect option `emit_person_boxes` (detector config): whole-body "person" boxes came out of
  the head fallback even when only faces were requested. Default unchanged (`true`) pending review,
  because on the reference recording those boxes were the only thing covering the hand-redacted area.
- Detector ensemble accepts an optional fine-tuned document/plate model (`doc_model`,
  `yolox_tiny_doc.onnx`) when the file is present. No such model ships yet: the fine-tune did not
  generalise (see `bench/results/REFERENCE_TANYARD.md`), so behaviour is unchanged.
- Bench tooling (developer only, nothing ships): ground truth from a hand-redacted reference copy
  (`labels_from_reference.py`), leak report, re-scoring from saved tracks, open-vocabulary
  teacher evaluation and pseudo-labelling, YOLOX fine-tuning loop for Windows, face identity
  grouping (`face_identities.py`).

### Fixed
- `BBox.to_list()` returns plain floats (numpy float32 from the detectors broke the JSON audit
  manifest and bench output).

## [0.1.12] - 2026-10-16

### Fixed (pre-user-testing review)
- Export asks before replacing an existing output file or its report/project/manifest sidecars.
- Batch processing never overwrites an existing `.vrproj` or `_REDACTED` file; it writes a numbered sibling.
- Space and the arrow keys now work after clicking the video, timeline or transcript (they were
  swallowed by those panels).
- Starting a transcription no longer wipes the existing transcript before the model has loaded, so a
  failed load (e.g. a model size that is not bundled) keeps the old text.
- Export and batch refuse to start while tracking/detection/transcription jobs are running.
- Deleting a track longer than a second asks for confirmation (there is no undo).
- Update notification showed raw HTML for the link.
- "Redact audio range" and the audio lane are disabled on files without an audio stream.
- Tools → "Check detection models…" and Settings show the bundled model folder; Quick guide lists all keys.

### Changed
- Transcription language defaults to English (`language: "en"`, editable in Settings; blank = auto-detect).
  On noisy body-cam openings auto-detect picked a random script and produced non-Latin garbage
  while running 4x slower (measured on two sample clips, 2026-10-16). Existing installs that saved
  the old blank value are migrated once (`settings_rev`).

## [0.1.11] - 2026-10-15

### Changed
- Update checker now only notifies: a background check of GitHub Releases at startup (5 s
  timeout, silent when offline) shows the new version with a link to the release page. The
  in-app download/install path was removed; nothing is installed automatically.
- Startup update check is on by default again (notification only, public metadata only); it can
  be turned off in Settings → Network. Model downloads remain disabled by default.

### Added
- `VERSION` file, this changelog, and `scripts/bump_version.py --check/--notes`; a test fails if
  the version copies or the changelog section are missing.
- GitHub Actions workflow (`.github/workflows/ci.yml`): tests on every push/PR; on each published
  release it also runs the benchmark on the bundled synthetic sample clip (never department
  footage) and attaches the results to the release.
- `docs/LICENSES.md`: licence of every library and model, with the non-permissive ones flagged.
- `bench/results/phase4.json` + `bench/results/COMPARISON.md`: benchmark after Phase 4.

### Fixed
- Export crashed when writing the audit manifest for auto-detected tracks (numpy float32
  coordinates were not JSON serialisable). Found by the Phase 4 benchmark re-run.

## [0.1.10] - 2026-10-15

Phase 4: evidence integrity and compliance.

### Added
- JSON audit manifest beside every export (`<output>_audit.json`): operator, timestamps, software
  version, machine, inference tier, SHA-256 of source (at open and at export), output, project and
  every model file with licence, thresholds, every redaction (frame ranges and boxes), every manual
  edit, verification result; self-hashed, hash printed in the PDF/CSV.
- Source re-hashed at export and compared with the open-time hash; mismatch flagged in the result
  dialog, the PDF and the project log.
- Model table in the PDF report.
- `--selftest --offline-check`: runs the full self-test with every socket blocked and reports any
  connection attempt.
- `docs/CJIS_COMPLIANCE.md`, `docs/PHASE4_NOTES.md`, `tests/test_audit.py`.

### Changed
- Pixelate enforces a 16 px minimum block; "blur" is now a heavy blur (about 6 samples across).
  Black remains the default. UI labels say which styles are recommended.
- Offline by default: no runtime model downloads (clear error instead), Hugging Face offline/telemetry
  opt-outs, ONNX Runtime telemetry events disabled, `openvino_telemetry` excluded from the installer
  and made un-importable (it posted to Google Analytics when no consent file existed).

## [0.1.9] - 2026-10-15

Phase 3: speed.

### Added
- Decode/encode worker threads with bounded queues (`core/streams.py`); FFmpeg proxy pipe with
  QuickSync for sources wider than 1280 px.
- OpenVINO runner on Intel iGPU/NPU with disk-cached compiled models; ONNX Runtime fallback.
  YuNet moved off OpenCV DNN onto ORT/OpenVINO with raw NHWC input.
- Scene-cut detection and coverage-gated adaptive detection stride; tracker pool; C++ mask
  compositing; network-share sources staged to local disk; batch queue (File → Batch process).
- Benchmarks: `bench/results/phase3_*.json`.

## [0.1.8] - 2026-10-14

Phase 2: accuracy.

### Added
- Pluggable detectors (`vision/detectors/`), tiled and multi-scale wrappers, head-fallback merged
  into the face lane, 25 % face padding with ellipse masks.
- BoT-SORT association (Roboflow `trackers`, Apache-2.0) kept additive to the raw detections;
  two-pass lead-in/lead-out back-fill and interpolation.
- Review timeline flags and keyboard review; `train/` fine-tune scaffold; `bench/` harness with
  leak-rate metric and local Label Studio workflow.
- Post-export verification: faces re-detected in the output and uncovered ones listed.

## [0.1.7] - 2026-10-06

### Fixed
- Updater skips the Restart Manager check (`/NOCLOSEAPPLICATIONS`), which hung for 20+ minutes.

## [0.1.6] - 2026-10-06

### Added
- Clip-range export ("Export only a time range"); report notes the range; self-test exports 20 s.

## [0.1.5] - 2026-10-06

### Changed
- ViTTrack default tracker; parallel carry-tracking auto-detect; post-export face verification;
  product research in `docs/RESEARCH-*.md` and a roadmap derived from it.

## [0.1.4] - 2026-10-06

### Fixed
- Qt combo boxes returned plain strings for style enums, crashing style changes and tracking
  start; `as_enum()` coercion everywhere with a regression test.

## [0.1.3] - 2026-10-06

### Changed
- First real-footage fixes: KCF tracker (about 10x faster than CSRT), ORT thread spinning off,
  VAD off by default with filler filtering, models and the Whisper `small` model bundled in the
  installer, background jobs panel, timeline drag-to-redact for audio.

## [0.1.2] - 2026-10-06

### Added
- GitHub Releases publishing; verified the in-app update path 0.1.1 → 0.1.2.

## [0.1.1] - 2026-10-06

### Added
- First installer build (Inno Setup, per-user or per-machine), `Published\` folder with
  `latest.json` and `SHA256SUMS.txt`, version bump on every build.
