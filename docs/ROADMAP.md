# Roadmap

Ordered roughly by value to an agency user. Tick items off and move them to
NOTES.md "Current state" when done. The research behind the ordering is in
docs/RESEARCH-commercial.md, RESEARCH-legal.md and RESEARCH-technical.md.

## Done in v0.1.5 (performance round)
- [x] ViTTrack default tracker (scale-adaptive, 5 ms/frame); KCF/CSRT selectable.
- [x] Auto-detect: carry tracks through missed detections, keep single detections, link fragments, parallel workers, full-res detection.
- [x] Post-export verification pass (re-detect faces in the output, list uncovered hits).

## Top 10 next, drawn from the commercial / legal / technical research
1. **Review queue** for auto-detections: list every track with a thumbnail, jump-to, approve/delete, "mark all reviewed", keyboard F / Shift+F to step through (Axon/Motorola pattern). Biggest reviewer time saver. (M)
2. **Exemption code + reason per redaction**, state pick-lists, exemption log in the PDF/CSV (FOIA 552(b), RCW 42.56.210). (S)
3. **Full-frame blackout / blur for a time range** with audio drop, one click (MDT screens, residence interiors). (S)
4. **Unique-person gallery**: cluster face tracks with SFace embeddings (OpenCV, permissive) so one click redacts a person everywhere; "redact everyone except the subject". (L)
5. **OpenVINO execution provider** for the detectors on Intel CPUs/iGPUs (1.3-4x). (S)
6. **License plate detector**: open-image-models YOLOv9-t (MIT) after checking training-data terms. (S)
7. **Audio: speaker diarization** via sherpa-onnx (mute one voice), **keyword spotting**, whisper large-v3-turbo as "accurate" mode, loudnorm pre-pass. (M)
8. **Axon-style shortcuts** (A/D frame step with hold, Q/E 2 s, W/S resize, [ ] trim to playhead, hold M for audio) + in/out trim and clip-only export. (S)
9. **Tamper-evident audit log** (hash-chained, Windows user) + review/approval state + per-request work timer (cost recovery). (M)
10. **Batch queue** with presets, overnight run, watch folder (CaseGuard pattern). (M)

## Next up
- [x] First QA on real body-cam footage (720p30, 14 + 47 min Axon clips) -> v0.1.3 speed/VAD/model fixes.
- [ ] Measure export time on a full 47-min clip; add ETA to job rows; consider hardware decode.
- [x] Windows installer + GitHub releases + auto-update; models bundled.
- [ ] Replace the GPL ffmpeg test binary with an LGPL build in `bin\` before wide distribution.
- [ ] Undo/redo for redaction edits (command stack over Project).
- [ ] Review queue: list low-confidence detections and track losses, jump through them.
- [ ] Transcript export (TXT/SRT with redacted words replaced by [REDACTED]).
- [ ] Waveform lane in the timeline; drag redaction edges to adjust (drag-to-create exists).
- [ ] Custom word lists (names, addresses) applied to every new transcript; fuzzy matching.
- [ ] Multi-object "redact every face" fast path: run auto-detect faces, then KCF between detections.

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
