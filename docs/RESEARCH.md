# Reference projects reviewed (2026-10-06)

Summary of the open-source projects Wyatt pointed at, what we took from each,
and what we must not take. Items marked *unverified* were not confirmed.

| Project | License | What it is | Use for us |
|---|---|---|---|
| seattle-police/redactvideo | none (all rights reserved) | 2015 Flask body-cam redaction web app | ideas only: bidirectional tracking, obscure+outline rendering |
| GGMuttaky/redactor | GPL-3.0 | local face anonymizer with review UI | ideas only: per-face on/off, keyframe interpolation, downsampled blur, audit report with checksums and no thumbnails |
| monjurulkarim/video-redactor | proprietary binaries | commercial desktop tool | ideas only: review queue (approve/exempt), draw-once-and-follow, honesty note that only solid fill is guaranteed |
| facebookresearch/EgoBlur | Apache-2.0 (click-through) | face + **license plate** detectors, TorchScript Faster R-CNN, ~400 MB each | too heavy for CPU full-video; candidate for an optional "deep pass" after ONNX export |
| ORB-HD/deface | MIT | CLI face anonymizer | **CenterFace ONNX (MIT)** adopted as optional high-recall face detector; preprocessing: resize to multiples of 32, no normalization, outputs heatmap/scale/offset/landmarks |
| facebookresearch/sam2 | Apache-2.0 | promptable video segmentation | future "precise mask" mode (sam2.1 tiny ONNX exists); CPU full-video propagation unrealistic (*unverified timing*) |
| IDEA-Research/Grounded-SAM-2 | Apache-2.0 | text-prompted detect + SAM2 track | CUDA-bound; pattern only |
| VolksRat71/sam-ui | Apache-2.0 | SAM2 video UI | UX: per-object lanes, re-track after correction, review queue of low-confidence frames, undo/redo |
| mikel-brostrom/boxmot | **AGPL-3.0** | MOT trackers + ReID | **do not import**. Re-implement from MIT originals: ByteTrack (ifzhang), OC-SORT (noahcao), BoT-SORT (NirAharon). StrongSORT is GPL: avoid |
| m-bain/whisperX | BSD-2 | alignment + diarization on top of whisper | not adopted: pulls full PyTorch + gated pyannote; faster-whisper already gives word timestamps. Revisit if speaker-based redaction is requested |
| microsoft/presidio | MIT | PII detection framework | planned optional NER backend (spaCy en_core_web_sm ~12 MB) behind our rule engine; needs spoken-form number normalization first |
| Arfa-Ahsan/PII_Detection... | none found | call-recording PII muter (LLM-based) | ideas only |
| cleanroom-ai/audio-pii-redaction | Apache-2.0 | in-browser redactor | adopted ideas: spoken-form digit normalization, +/-80 ms padding (ours: 50 ms default, configurable), SRT/TXT export (todo) |
| neonwatty/bleep-that-shit | Apache-2.0 | browser word bleeper | adopted: exact/phrase matching, selectable bleep styles, preview before export; todo: fuzzy matching, word lists |
| Songinpyo/Open-Face-Blur | MIT | manual face blur GUI (MTCNN) | keyboard shortcuts idea; MTCNN too slow on CPU |

## Model decisions

- Faces: **YuNet** (`face_detection_yunet_2023mar.onnx`, MIT, OpenCV zoo) primary via
  `cv2.FaceDetectorYN`; **CenterFace** (MIT) registered as optional higher-recall detector (not wired into the UI yet).
- Objects: **YOLOX-s / YOLOX-tiny** ONNX (Apache-2.0) for COCO classes mapped to
  screen (tv, laptop), phone, document (book), person, keyboard, mouse.
- License plates: **no permissive CPU-light model found.** Options: (a) export
  EgoBlur LP to ONNX and offer as slow optional pass; (b) train a small
  Apache-licensed detector (RT-DETR / NanoDet) on an open plate dataset;
  (c) rely on draw-and-track (works today).
- Caveat flagged by redactor's README (*unverified*): WIDER FACE training data
  may carry a non-commercial license, which could affect YuNet/CenterFace
  weights. Worth a legal look before wide distribution; since VideoRedact is
  free and non-commercial this is likely fine but should be documented.

## Techniques adopted

- Detect on downscaled frames, output at full resolution (deface).
- Keyframes + interpolation; lead-in/lead-out padding (redactor).
- Audit report with SHA-256 of source and output, no PII thumbnails (redactor).
- Solid fill described as the only guaranteed mode (video-redactor).
- Audio padding around word boundaries; bleep/silence modes (cleanroom-ai, bleep-that-shit).
