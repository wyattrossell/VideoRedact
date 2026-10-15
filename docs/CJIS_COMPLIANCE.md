# VideoRedact: evidence integrity and CJIS compliance statement

Written for an agency's CJIS compliance file and for the person who has to answer
"what does this program do with our footage?". Everything below can be verified on the
installed program with the commands in the last section. Applies to VideoRedact 0.1.10
and later.

## 1. Summary

| Requirement | How VideoRedact meets it |
|---|---|
| Original evidence is never altered | The program only reads the source file. Output is always a new file. The source SHA-256 is taken at open and re-computed at export; the two values and whether they match are recorded in the audit manifest, the PDF/CSV report and the project file. |
| Chain of custody for every export | A JSON audit manifest (`<output>_audit.json`) is written beside every export with operator, timestamps, software version, machine, model hashes, thresholds, every redaction (frame ranges and boxes) and every manual edit. The manifest carries its own SHA-256, which is also printed in the PDF report. |
| Redaction is irreversible | Default style is a solid black fill. Pixelate uses blocks of at least 16 output pixels (each block is the mean colour of its area). "Blur" is a heavy blur: the region is averaged down to about 6 samples across before smoothing. Light Gaussian blur was removed because published attacks recover faces and text from it. |
| Audio is preserved by default | Audio is copied from the source unless the operator adds audio redactions (beep / silence / low tone / noise over a time range). Audio redactions are listed in the manifest. |
| Fully offline | No telemetry, no analytics, no cloud calls, no runtime model downloads. All models ship inside the installer. The one optional network feature (checking GitHub for a newer installer) is off by default. Verified by a built-in test that blocks every socket and runs the full pipeline (section 6). |
| Revisable | Every export also writes a `.vrproj` project file. Reopening it restores the transcript, every redaction and the audit log so a redaction can be revised without starting over. |
| Licensing | MIT application; all bundled models and libraries are MIT/Apache/BSD (see the model table in every PDF report). No GPL/AGPL components in the application code. FFmpeg build licence is noted in `NOTES.md`. |

## 2. Network behaviour

VideoRedact makes **no network connections** in normal operation. The complete list of
code paths that *can* open a connection, and how each is controlled:

| Code path | Destination | When it runs | Default |
|---|---|---|---|
| Update check (`videoredact/updater.py`) | `https://api.github.com/repos/wyattrossell/VideoRedact/releases/latest`, then `https://github.com/.../releases/download/...` for the installer | Only when **Settings → Network → Check for updates** is on (at startup) or the operator chooses **Help → Check for updates** | **Off** |
| Model download (`videoredact/vision/models.py`) | `github.com` (OpenCV Zoo, Megvii YOLOX release assets) | Only when a model file is missing **and** offline mode is off | Models are bundled; **offline mode on**, so a missing model is an error, not a download |
| Speech model (`videoredact/audio/transcribe.py`, faster-whisper / huggingface_hub) | `huggingface.co` | Only when the speech model folder is missing **and** offline mode is off | Bundled; loaded with `local_files_only=True`; `HF_HUB_OFFLINE=1` set process-wide |

Nothing else in the program imports a network library. Auto-detection, tracking,
transcription, export and reporting run entirely on the local machine.

### Third-party telemetry, and what was done about it

| Library | Telemetry behaviour as shipped by the vendor | VideoRedact control |
|---|---|---|
| **OpenVINO** (Intel iGPU/NPU inference) | `import openvino` loads its model-conversion tool, which initialises the separate `openvino_telemetry` package. With no consent file on the machine that package treats the user as **opted in** and posts usage events to `www.google-analytics.com`. (Found by our socket-blocking self-test on 2026-10-15.) | The `openvino_telemetry` package is **excluded from the installer** and additionally made un-importable in every process (`videoredact/offline.py`), so OpenVINO falls back to its built-in no-op telemetry stub. `OPENVINO_TELEMETRY_OPT_IN=0` is also set. |
| **ONNX Runtime** | Windows ETW "telemetry events" (local Event Tracing, not network). | `onnxruntime.disable_telemetry_events()` is called before any session is created. |
| **huggingface_hub** (dependency of faster-whisper) | Can send usage headers and check the Hub when loading a model. | `HF_HUB_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`; model loaded `local_files_only`. |
| **Ultralytics / YOLO** | Settings sync and analytics. | **Not shipped** (AGPL). `YOLO_OFFLINE=1` is set anyway as a documented guard. |
| PySide6 / Qt, OpenCV, reportlab, Roboflow `trackers`, `supervision`, CTranslate2 | No telemetry. | `DISABLE_TELEMETRY=1`, `DO_NOT_TRACK=1` set for any library that honours the generic opt-outs. |

The environment variables above are set in `videoredact/offline.py`, which runs when the
`videoredact` package is imported, i.e. before any third-party library loads, in the GUI,
in the command-line self-test and in every worker process.

## 3. Where data is written

| Data | Location | Notes |
|---|---|---|
| Redacted output, PDF/CSV report, `.vrproj` project, `_audit.json` manifest, `_verification.csv` | The folder the operator chooses at export | These are the evidence products; retain per your policy. |
| Settings | `%LOCALAPPDATA%\VideoRedact\settings.json` | No case data. |
| Logs | `%LOCALAPPDATA%\VideoRedact\logs\` | File names and timings of operations; no frames, no transcript text. |
| Models | `<install dir>\_internal\models\` (bundled, read-only) and `%LOCALAPPDATA%\VideoRedact\models\` (user copies) | OpenVINO compiled-kernel cache under `models\cache\openvino` holds model code only. |
| Speech model | `<install dir>\_internal\models\whisper\` | |
| Staging copy of a source that lives on a network share | `%TEMP%\videoredact_stage\` | A byte-identical copy made for speed (hash-verified). Up to 20 GB kept; oldest removed first. Set `VIDEOREDACT_NO_STAGING=1` to disable. Clear the folder to purge. |
| Transient | `%TEMP%\videoredact_*` (extracted audio WAV, per-segment encode files, self-test output) | Deleted at the end of each operation; the self-test output is overwritten each run. |

No data leaves the machine. Nothing is written to the source file's folder unless the
operator chooses it as the output folder.

## 4. Hashing and the audit manifest

* **Algorithm:** SHA-256 (hashlib), full file, computed in the program; the same value
  `certutil -hashfile <file> SHA256` or `Get-FileHash` produces.
* **Source:** hashed when the file is opened (stored in the project), re-hashed at export.
  `source.unchanged_since_open` in the manifest is `true`, `false` (the PDF then carries a
  warning and the project log an entry `WARNING_source_hash_changed`) or `null` (file opened
  without hashing, e.g. a project created by an older version).
* **Output:** hashed after the encoder closes the file.
* **Project file:** hashed after it is saved; the manifest records that hash, so the
  `.vrproj` on disk is tied to the manifest.
* **Model files:** every model file present is hashed and listed with its licence, so a
  reviewer can tell exactly which detector produced the redactions.
* **Manifest:** `manifest_sha256` is the SHA-256 of the manifest's canonical JSON body
  (sorted keys, no whitespace, excluding the hash field). It is printed in the PDF report and
  the CSV. `python -c "from videoredact.core.audit import verify_manifest; print(verify_manifest('x_audit.json'))"`
  recomputes it; any edit to the manifest after export makes it fail.

Manifest sections (`format: videoredact-audit`, `format_version: 1`): `software`,
`generated_at` (local time with UTC offset), `operator` (name entered in the project, Windows
user, machine, OS), `case_number`, `notes`, `source`, `output`, `export_options` (codec,
quality, clip range, verification labels…), `engine` (inference tier and library versions),
`models`, `transcription_model`, `settings` (every detection threshold and option in force),
`redactions.audio[]` (id, start/end seconds, words covered, style, how it was created),
`redactions.video[]` (id, label, style, shape, padding, every span's frame range and every
keyframe box in normalised coordinates), `verification` (post-export re-detection result:
frames checked, detections, any uncovered boxes), `audit_log[]` (every manual edit, with
timestamp and Windows user, from the moment the file was opened).

## 5. Redaction irreversibility

* **Black (default):** pixels are replaced by zeros. Nothing to recover.
* **Pixelate:** block size = max(16 px, short side of the box / 8). Each block is replaced by
  the mean colour of its area (area-average downscale, nearest-neighbour upscale). A 30 px
  face becomes a 2×2 grid of flat colours. Published "de-pixelation" work needs far smaller
  blocks relative to the face; the 16 px floor is enforced in code (`PIXELATE_MIN_BLOCK` in
  `videoredact/core/video_redact.py`) and tested (`tests/test_audit.py`).
* **Heavy blur:** the region is area-averaged down to about 6 samples across, smoothed, and
  scaled back up. Visually a soft blur, informationally the same as a 6-block pixelation. The
  earlier adjustable Gaussian blur was removed. The UI labels this style "not for text"
  because the smooth result can give a false impression that detail was merely softened.
* Styles are chosen per project and per item; the project file and manifest record which was
  used for every track.
* **Verification pass:** after export the program can re-run the face detector on the
  *output* and report any face it still finds that is not covered (`_verification.csv`,
  summarised in the manifest and PDF). Enabled by the "Verify export" option.

## 6. How to verify on an installed machine

1. **No network:** in a command prompt
   ```
   "C:\Program Files\VideoRedact\VideoRedact.exe" --selftest --offline-check "C:\path\to\any.mp4"
   ```
   Every socket `connect()` is replaced with one that raises and records the target. The full
   pipeline (detectors on the configured hardware tier, speech model load, export, report)
   runs. The last line reads `offline check: 0 blocked connection attempt(s) []` and the exit
   code is 0. Any attempt lists the host and port and the exit code is 2. (Run from a
   network-monitoring tool such as Windows Resource Monitor or a firewall log if an
   independent confirmation is wanted; the program shows no connections.)
2. **Settings:** `%LOCALAPPDATA%\VideoRedact\settings.json` should contain
   `"offline": true` and `"check_updates": false` (both are the defaults; absent keys mean
   default).
3. **Export integrity:** export a file; confirm `<output>_audit.json` exists, that the hash in
   the PDF report's "Audit manifest" line equals `manifest_sha256` in the file, and that
   `Get-FileHash` on the output and the source equal the values in the manifest.
4. **Source untouched:** compare the source file's hash and modified time before and after an
   export.
5. **Model inventory:** the PDF report's "Detection / recognition models" table lists each
   model file with its SHA-256; compare with the files under `<install dir>\_internal\models`.

## 7. Retention and handling recommendations

* Treat the output video, `.vrproj`, `_audit.json` and the PDF/CSV as one evidence package;
  store them together.
* The `.vrproj` contains the full transcript text and every box; it is as sensitive as the
  source. Store under the same controls.
* Clear `%TEMP%\videoredact_stage` when a workstation is decommissioned or re-assigned.
* Keep the installer that produced an export (version is in the manifest) so a redaction can
  be reproduced.

## 8. Known limitations / honest notes

* The operator name in the manifest is whatever was typed into the project's "Author" field
  plus the Windows login; there is no separate authentication inside the program.
* The manifest proves what the software did; it does not sign it. A hash chain into a
  write-once evidence system is the agency's responsibility.
* Update checking, when an operator turns it on, sends the standard HTTPS request headers
  (including a `VideoRedact/<version>` user agent) to GitHub. Nothing about the case is sent.
* Transcription (speech-to-text) runs locally; its accuracy on body-camera audio is limited
  and the transcript should not be treated as a verbatim record.
