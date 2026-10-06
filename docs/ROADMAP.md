# Roadmap

Ordered roughly by value to an agency user. Tick items off and move them to
NOTES.md "Current state" when done.

## Next up
- [ ] QA on real body-cam / interview-room / dash-cam footage (1080p, 30-60 min).
      Measure detect + export times on an 8-core box; tune `detect_stride`,
      YOLOX-tiny default for long clips, progress ETA.
- [ ] Build the Windows installer once (`installer\build.ps1`), fix PyInstaller
      hidden-import gaps (onnxruntime, ctranslate2, PySide6 multimedia plugins),
      bundle LGPL FFmpeg and the detection models + whisper `small`.
- [ ] Undo/redo for redaction edits (command stack over Project).
- [ ] Review queue: list low-confidence detections and track losses, jump through them.
- [ ] Transcript export (TXT/SRT with redacted words replaced by [REDACTED]).
- [ ] Waveform lane in the timeline; drag redaction edges to adjust.
- [ ] Custom word lists (names, addresses) applied to every new transcript; fuzzy matching.

## Detection and tracking
- [ ] License plates: evaluate EgoBlur LP -> ONNX as optional deep pass; or train a small Apache-licensed detector.
- [ ] Wire CenterFace as selectable face detector (higher recall on small/profile faces).
- [ ] Periodic re-detection inside a track to correct CSRT scale drift.
- [ ] Appearance embedding (OSNet-style ONNX, permissive) for stronger re-identification after long absences.
- [ ] Optional SAM2-tiny "precise mask" for a single object over a short range (benchmark CPU first).
- [ ] Person-level redaction (full body) when faces are turned away.

## Audio
- [ ] Optional Presidio + spaCy small model for names (behind our rule engine), with spoken-number normalization.
- [ ] Speaker diarization (redact everything a given speaker says) if agencies ask for it.
- [ ] Larger whisper models with RAM check; language auto-detect UI feedback.

## Evidence / workflow
- [ ] Report: include frame thumbnails *with redaction applied* as optional appendix.
- [ ] Batch mode: apply the same word list / detector settings to a folder of files.
- [ ] Verify output hash on reopen; warn if project media hash changed.
- [ ] Keyboard-first review mode (J/K/L shuttle, I/O marks).

## Packaging
- [ ] Code-sign the installer (agencies' AV will flag unsigned PyInstaller EXEs).
- [ ] Offline model bundle variant of the installer vs. download-on-first-run variant.
- [ ] Auto-update check (opt-in, GitHub releases).
