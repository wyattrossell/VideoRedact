# VideoRedact: guide for the first round of user testing

For officers and records staff trying VideoRedact 0.1.11 on real cases. Nothing you do in the
program touches the original video; every export is a new file. Keep working copies of
footage where your evidence policy says they belong.

## Install

1. Get `VideoRedact-<version>-Setup.exe` from https://github.com/wyattrossell/VideoRedact/releases
   (or from the shared `Published` folder). About 700 MB: all detection and speech models are inside.
2. Run it. No administrator rights are needed; it installs under your profile
   (`%LocalAppData%\Programs\VideoRedact`). An admin can choose "all users" instead.
3. First launch takes 10 to 20 seconds while the speech and detection models load. A few seconds
   later the program may tell you a newer version exists, with a link; it never installs anything itself.
4. Optional check that everything works on your machine (prints a report, exports a 20 s test clip):
   ```
   "%LocalAppData%\Programs\VideoRedact\VideoRedact.exe" --selftest "C:\path\to\any.mp4"
   ```
   Add `--offline-check` to prove it makes no network connections.

Hardware: made for an 8-core desktop with 16 GB and no graphics card. Intel machines with an
integrated GPU get a speed-up automatically (the self-test prints "tier=intel"). Expect about
3 to 4 minutes per 10 minutes of 720p body-cam footage for auto-detect and about 1 minute per
10 minutes for export on that class of machine.

## What to try, in order

1. **Open** a body-cam file (File → Open). The player, timeline and transcript panel appear.
   Help → Quick guide (F1) summarises the workflow inside the program.
   Playback with audio should be smooth; frame stepping with the arrow keys should be exact.
2. **Transcribe** (Audio → Transcribe speech). Progress shows in the Jobs panel at the bottom; text
   appears as it is produced. Expect 2 to 4 minutes for a 10-minute clip.
3. **Redact words**: click a word, or drag across a phrase, right-click → Redact. Search a
   name and use "Redact all matches". Try Audio → Suggest sensitive speech (PII) for phone
   numbers, dates of birth, addresses.
4. **Draw a box** on something (drag on the picture); in the dialog that opens choose *Track this
   object*. (Video → Track selected box re-tracks an existing box from the current frame.)
   Watch the box follow the object, disappear when it leaves the frame, and come back when it
   re-enters. Check the timeline lane shows the visible ranges.
5. **Auto-detect** (Video → Auto-detect faces/screens/documents). Review the list in the
   right-hand panel; use the review keys shown in the panel to step through tracks, delete
   false ones, extend short ones.
6. **Styles**: change the project default (toolbar) and one item's style (panel). Black box is
   the default and the recommended one; pixelate and heavy blur are there for footage where a
   solid box hides context.
7. **Audio redaction by range**: drag on the audio lane of the timeline to beep or silence a
   range without a transcript.
8. **Export** (File → Export redacted media, Ctrl+E). Leave "Verify" on: the program re-detects faces in the finished
   file and lists any it still sees. Output files beside the video:
   - `<name>_REDACTED.mp4` (or your chosen name), SHA-256 in the report
   - `<name>_redaction_report.pdf` and `.csv`
   - `<name>.vrproj` (reopen to continue editing)
   - `<name>_audit.json` (the audit manifest; keep it with the export)
   - `<name>_verification.csv` only if the verifier found uncovered faces
9. **Reopen the project** (File → Open project or double-click the `.vrproj`) and change
   something; export again.
10. **Batch** (File → Batch process files, Ctrl+B) on a folder of two or three short clips with the same
    settings.

## What to write down when something is wrong

- What you clicked, what you expected, what happened. A screenshot helps.
- The clip's length and resolution (the report's "Video" line has it). We cannot receive the
  footage itself; describe it instead (daylight/night, how many people, body-cam or dash-cam).
- The log file: `%LocalAppData%\VideoRedact\logs\` (newest file). It contains file names and
  timings, no video or transcript text.
- If the program closes unexpectedly, the same log has the error.

Send to Wyatt Rossell, or open an issue at https://github.com/wyattrossell/VideoRedact/issues
without attaching footage.

## Known limitations in this round

- **Accuracy numbers are not yet measured** on labelled footage. Auto-detect misses small or
  turned-away faces at times; always review the list and use the verifier. Report what it missed.
- **No licence-plate detector yet.** Draw a box and track it.
- **Transcription** is a local speech model; on noisy audio it is a guide for finding words,
  not a verbatim record. Voice activity detection is off by default because it dropped speech
  on body-cam audio.
- **No undo/redo.** Deleting a long track asks first; prefer unticking a redaction's checkbox to
   keep it in the project. Save the project often (Ctrl+S).
- **Large files** on a network share are copied to the local disk first (up to 6 GB per file);
  the first operation on such a file is slower.
- **Update check** only tells you a new version exists; install it from the releases page.
- Machines with an NVIDIA card are not faster yet (CPU/Intel GPU paths only).
- The bundled FFmpeg is a GPL build; we will swap it for an LGPL build before distributing
  beyond HPD (see docs/LICENSES.md). This does not affect testing.

## What testers should not do

- Do not point the program at original evidence files if your policy requires working copies.
- Do not upload footage, projects, reports or manifests anywhere to report a problem.
