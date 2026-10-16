# Changelog

All notable changes to VideoRedact. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/). The section for a version is used
verbatim as its GitHub release notes (`scripts/bump_version.py --notes`). Write entries under
**Unreleased** as you go; `installer/build.ps1` promotes them when it bumps the version.

## [Unreleased]

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
