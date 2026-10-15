# Phase 4: evidence integrity and compliance (v0.1.10, 2026-10-15)

What the user asked for and where each item landed. The agency-facing statement is
`docs/CJIS_COMPLIANCE.md`; this file is the engineering record.

| Requirement | Implementation | Verified by |
|---|---|---|
| Never modify the original | Unchanged (read-only pipeline). Now proven per export: source re-hashed at export, compared with the open-time hash, mismatch logged as `WARNING_source_hash_changed` and shown in the result dialog / PDF. | `tests/test_audit.py::test_source_untouched_and_rehashed` (hash + mtime unchanged) |
| SHA-256 of input and output | Already present; added `source_sha256_at_export`, `source_unchanged`, project-file hash, model-file hashes. | same |
| Sidecar audit JSON per export | `videoredact/core/audit.py`: `build_manifest` / `write_manifest` / `verify_manifest`. Written as `<output>_audit.json` by `core/export.py` (`ExportOptions.write_audit`, default on). Self-hash in `manifest_sha256`, printed in PDF and CSV. Order inside export: verify → save project → manifest (hashes the project) → PDF/CSV. The `audit_manifest` log line stays in memory so the saved project matches its hash. | `test_manifest_written_and_self_verifying`, `test_report_mentions_manifest`, tamper test |
| Re-editable project file | `.vrproj` already; now hashed into the manifest. | existing round-trip test |
| Irreversible redaction | `core/video_redact.py`: `PIXELATE_MIN_BLOCK = 16` floor; blur rewritten as area-average to ≤6 samples across + 3×3 smooth (`BLUR_DOWNSCALE = 6`). Default stays BLACK. UI labels: "Black box (recommended)", "Heavy blur (not for text)", "Pixelate (16 px+ blocks)". Combo order unchanged (BLACK, BLUR, PIXELATE) so saved projects and the UI test keep working. | `test_pixelate_has_minimum_block_size`, `test_blur_destroys_detail` |
| Preserve audio, hooks for audio redaction | Already true: audio copied unless ranges are added; beep/silence/tone/noise on timeline ranges. Manifest lists `redactions.audio[]`. | existing tests |
| Fully offline | `videoredact/offline.py` runs on package import in every process: env opt-outs, `openvino_telemetry` made un-importable. `paths.DEFAULT_SETTINGS`: `offline=True`, `check_updates=False`. `models.ensure_model` and `Transcriber` raise a clear error instead of downloading when offline. `main_window` startup check defaults off. `detector._ort_session` calls `ort.disable_telemetry_events()`. Installer spec excludes `openvino_telemetry`. `scripts/fetch_models.py` is the one place that lifts the guard (its own process only). | `--selftest --offline-check`, `test_ensure_model_refuses_download_when_offline`, `test_openvino_telemetry_is_unimportable`, `test_installer_excludes_telemetry_package`, `test_no_network_guard_blocks_sockets` |
| Document for CJIS file | `docs/CJIS_COMPLIANCE.md` | reviewed against the code paths above |

## The finding that justified the socket guard

The first run of `--selftest --offline-check` on the dev machine reported one blocked
connection: `('www.google-analytics.com', 443)`. Cause: `import openvino` imports
`openvino.tools.ovc`, which initialises `openvino_telemetry`; with no consent file in
`%LOCALAPPDATA%\Intel Corporation\` that package's `enable_opt_in_dialog=False` path treats
"no file" as consent and posts a GA4 event. Setting `OPENVINO_TELEMETRY_OPT_IN=0` did not
stop it (the package does not read that variable). Making the module un-importable makes
ovc use its own `telemetry_stub`. Second run: 0 attempts, exit 0.

Lesson recorded in NOTES.md gotchas: every new dependency gets run under
`--offline-check` before it ships.

## The `--offline-check` guard

`videoredact/app.py::_NoNetwork` monkey-patches `socket.socket.connect`,
`socket.socket.connect_ex` and `socket.create_connection` for the duration of the
self-test. It is deliberately simple: it catches every Python-level connection, including
urllib, requests, huggingface_hub and the telemetry package. It cannot see connections made
by native code that bypasses Python sockets (none of our native libraries, FFmpeg included,
open sockets in the code paths we call; FFmpeg is invoked with local file arguments only). An
agency wanting an independent check can run the self-test under a firewall log; the doc says
so.

## Settings added

| Key | Default | Meaning |
|---|---|---|
| `offline` | `true` | refuse runtime model downloads; sets `VIDEOREDACT_OFFLINE=1` for the process |
| `check_updates` | `false` (was `true`) | GitHub Releases check at startup; Help menu check still works on demand |

Settings dialog group renamed "Network (everything else in this program is offline)" with
both toggles.

## Not done / future

* Manifest signing (HMAC or certificate) if an agency's evidence system wants it. Hook:
  `build_manifest` returns the dict; signing would wrap `write_manifest`.
* A "verify package" command (`--verify <output>`) that re-hashes output/project/manifest
  and prints a pass/fail; today it is three manual steps in the compliance doc.
* LGPL FFmpeg build swap before wide distribution (still open from Phase 1).
