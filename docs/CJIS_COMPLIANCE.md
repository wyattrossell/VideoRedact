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
| Redaction is irreversible | Every style offered is a solid fill: black (default) or blue. Pixelate and heavy blur were withdrawn from the program in 0.1.19 (2026-11-02) after the re-identification bench (section 9) measured both weaker than a solid box; a project that used them is converted to black when opened, with an audit-log entry. |
| Audio is preserved by default | Audio is copied from the source unless the operator adds audio redactions (beep / silence / low tone / noise over a time range). Audio redactions are listed in the manifest. |
| Fully offline | No telemetry, no analytics, no cloud calls, no runtime model downloads. All models ship inside the installer. The one network feature, the startup check of GitHub Releases, requests only the public version number, shows a notification with a link, never installs, and can be turned off in Settings. Verified by a built-in test that blocks every socket and runs the full pipeline (section 6). |
| Revisable | Every export also writes a `.vrproj` project file. Reopening it restores the transcript, every redaction and the audit log so a redaction can be revised without starting over. |
| Licensing | MIT application; all bundled models and libraries are MIT/Apache/BSD (see the model table in every PDF report). No GPL/AGPL components in the application code. FFmpeg build licence is noted in `NOTES.md`. |

## 2. Network behaviour

VideoRedact makes **no network connections** in normal operation. The complete list of
code paths that *can* open a connection, and how each is controlled:

| Code path | Destination | When it runs | Default |
|---|---|---|---|
| Update check (`videoredact/updater.py`) | `https://api.github.com/repos/wyattrossell/VideoRedact/releases/latest` (public release metadata: version number, notes). Nothing is downloaded; the user gets a notification with a link to the release page. | A few seconds after startup when **Settings → Network → Check for updates** is on, or when the operator chooses **Help → Check for updates**. Background thread, 5 s timeout, silent when offline. | **On** (notify only). Turn it off in Settings for air-gapped or policy-restricted workstations; no other code path contacts the network. |
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
  reviewer can tell exactly which detector produced the redactions. Since 0.1.18 the registry
  carries the published SHA-256 of every bundled model; the manifest's `models[].matches_registry`
  says whether the file used is that file, downloads that differ are rejected, and
  `--selftest` has a "model integrity" step.
* **Output metadata:** the output is assembled from a raw frame pipe and the program's own
  audio, so no container or stream tag of the source (camera model and serial, GPS, creation
  time, comments) is ever copied into it. The manifest records every tag the output does carry
  (`output.metadata_tags`) and the result of checking the source's tag values against them
  (`output.source_metadata_carried`, expected `[]`); a non-empty list is also logged as
  `WARNING_source_metadata_in_output`.
* **Atomic write:** the encoder writes `<name>.part.<ext>` and the file is renamed to the
  output name only when it is complete, so an interrupted export never leaves a half-written
  file under the released name.
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
keyframe box in normalised coordinates), `redactions.protected_zones[]` (regions that are never
redacted whatever covers them, by default the camera's timestamp overlay; id, box, enabled),
`verification` (post-export re-detection result:
frames checked, detections, any uncovered boxes), `audit_log[]` (every manual edit, with
timestamp and Windows user, from the moment the file was opened).

## 5. Redaction irreversibility

* **Black (default):** pixels are replaced by zeros. Nothing to recover.
* **Blue:** pixels are replaced by one constant colour (RGB 20, 70, 180). Nothing to recover;
  it exists for agencies whose release format marks redactions in a colour other than black.
* **Pixelate and heavy blur: withdrawn in 0.1.19 (2026-11-02).** Both were irreversible by
  construction (16 px block floor; 6-sample averaging) but the re-identification attack in
  section 9 still re-identified a 117 px face under either at the loose threshold, while black
  re-identified none. The program now offers solid fills only. The code paths remain so the bench
  can keep measuring them (`bench/reid_attack.py`) and so an old project opens: its blurred or
  pixelated tracks become black boxes and the project's audit log records the change
  (`video_style_retired`).
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
   `"offline": true` (the default; absent keys mean default). `"check_updates"` defaults to
   `true` (notify-only); set it to `false` on workstations that must make no outbound
   connection at all.
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
* Update checking (on by default, notification only) sends the standard HTTPS request headers
  (including a `VideoRedact/<version>` user agent) to GitHub. Nothing about the case is sent.
* Transcription (speech-to-text) runs locally; its accuracy on body-camera audio is limited
  and the transcript should not be treated as a verbatim record.
* A **protected zone** (0.1.19; by default over the Axon timestamp overlay, top right) is never
  redacted, whatever track covers it, so the timestamp stays readable in a release. A real face
  that passes behind the zone is therefore left visible by design; the export verification
  re-detects faces in the output and lists it, and the zone is recorded in the manifest, the
  report and the project's audit log. Turn the zone off (Video menu) when the footage has no
  overlay there, or drag it to fit another camera.

## 9. Measured redaction strength, and how to quote it

`bench/reid_attack.py` attacks a redacted export the way an adversary holding the original
would: the face recogniser (SFace) is aligned on the *original's* landmarks and run on the
*redacted* pixels, and the result is compared with the original identity. Measured on
2026-10-31 on one 45-second 720p body-cam clip (crash scene, 45 face tracks, 80 confirmed
face-frames covered by the shipped auto-detect profile; control: the recogniser matched
unredacted faces of the same person at 0.73 mean similarity against 0.23 between different
people, so the clip supports the measurement):

| Style | Re-identified at the app's same-person threshold (0.363) | At 0.55 | Still detected as a face | Mean similarity after redaction |
|---|---|---|---|---|
| Black (default) | 0 of 80 | 0 | 0 | -0.02 |
| Pixelate (16 px floor) | 3 of 80 | 0 | 0 | 0.17 |
| Heavy blur | 4 of 80 | 0 | 1 | 0.14 |

Mean similarity after redaction sits below the different-person level in every style, i.e. a
redacted face matches its own original no better than a stranger would. The residual matches
at the loose threshold are all in blur and pixelate; the numbers and the faces behind them are
in bench/results/REID_ATTACK.md. On these numbers blur and pixelate were withdrawn from the
program in 0.1.19; the rows stay here because they are the reason. Black is the default; blue
is the same solid fill in another colour (not re-measured: identical construction).

Quoting these numbers (rules adopted from OpenScrub's benchmark notes):

* Always state the version, the style, the clip description and the hardware. A number without
  them is not a claim.
* A measured 0 is a result on that footage, not a promise about footage nobody has run.
  Say "0 of 80 face-frames on the crash-scene clip were re-identified", never "faces cannot be
  re-identified".
* Never describe the program as "CJIS compliant", "court certified" or "guaranteed". Tools are
  not compliant; an agency's process is.
* Publish the failures next to the successes (the blur and pixelate rows above are the point).
