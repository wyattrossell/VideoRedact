# Reference projects reviewed (2026-10-31)

Four more repositories Wyatt pointed at, read in full (cloned locally, nothing executed). All
four are Apache-2.0, so code could be reused with attribution; in practice everything below was
re-implemented on our own detector, track and audit models, and the attribution is recorded in
docs/LICENSES.md section 5. The judgement throughout: what helps a body-cam reviewer on an
8-core CPU machine with no GPU, under the precision policy of 2026-10-28.

| Project | What it is | Taken | Rejected / not applicable |
|---|---|---|---|
| austinmabry/OpenScrub (Python, ~8k lines, web UI) | Local video redaction aimed at screen recordings: OCR text PII, faces, plates (download-on-demand), full-body silhouettes (AGPL YOLO-seg, download-only), scroll tracking, review cards per person, HDR, audit report, re-identification benchmark | **Edge assist** (mirror-pad the frame so faces cut off by the border are detected; off by default here, see below). **Temporal cannot-link** for same-person matching (tracks on screen together are different people). **Re-identification attack** as a benchmark (`bench/reid_attack.py`). Per-shape compile cache in the OpenVINO runner (their alternating-pass pattern exposed our single-slot cache). Rules for quoting benchmark numbers (docs/CJIS_COMPLIANCE.md section 9). | OCR/NER text redaction and scroll tracking (screen recordings, not body cams; OCR on 720p body-cam text is unreliable and Tesseract/Paddle add 100s of MB). Silhouette masking (AGPL weights). SCRFD registry entry (non-commercial weights). Download-on-demand model registries (we ship bundled, offline). Inpaint mode (reconstructs pixels; weaker than black for evidence). Head boxes from person detections (we have it; Wyatt turned it off for precision). Encryption-at-rest vault (agencies use BitLocker; a lost password destroys evidence). |
| ssrajadh/sentryblur (Python, ~1.5k lines, CLI) | One-command face/plate blur per frame, no tracking; SAM 3 "prompt" mode (GPU only, gated weights) | **Atomic output write** (encode to `<name>.part.mp4`, rename when complete; a crash or cancel never leaves a half-written file under the final name). Contact-sheet preview idea already covered by our live preview and bench sheets. | Per-frame detection with a ±3-frame mask union instead of tracking (our tracks already interpolate every frame; the union would widen boxes on fast motion only). SCRFD/InsightFace faces (non-commercial). open-image-models YOLOv9 plates (see PLATES.md: training provenance). SAM 3 prompt mode (GPU, bespoke licence, 25x real time). |
| ak375456/privacycam (Flutter, iOS/Android) | Phone app: faces, people, text, codes, plates on photos, short videos and PDFs; originals never overwritten; exports stripped of metadata and verified | **Metadata proof at export**: the output's container/stream tags are recorded in the audit manifest together with a check that none of the source's tag values (device, GPS, creation time, comments) reappear; verified by test. Their plate post-filters (aspect ratio 1.1-8, 10/20 % safety padding, per-frame cap) are noted for the day a plate detector ships. | Google ML Kit (cloud SDK diagnostics), 60-second video cap, "pro" tier. Their track builder (exit guard with motion lead, boundary envelope) is a cruder version of our carry tracking; the motion-lead idea is parked until `track_eval` shows a hold-box gap. Edit scopes "this moment / from here / entire track" are a UI question for Wyatt. |
| caipeijia833/doc-redaction (Java, Chinese office documents) | Office/PDF/OFD/image/audio/video redaction server bound to localhost; 205 rules; AES-GCM restore vault; Ed25519-signed offline updates; hash-verified components at startup; fail-closed philosophy | **Model integrity check**: registry SHA-256 for every bundled model, verified after download and by `--selftest` ("model integrity"); the audit manifest records whether each model used matches the published file. | Document formats, OCR, Chinese rules (out of scope). Restore vault (we never modify originals). Their sampled-frame "temporal bridge" is weaker than our carry tracking. Ed25519 manifest signing is a design question for Wyatt (key custody), not something to build blind. |

**Edge assist, measured** (`EdgeAssist` in vision/detectors/wrappers.py; scratch script, 990
sampled frames of the 11 Axon bench clips, YuNet@1280 + multiscale with and without the pad):
the first design (padded pass replaces the normal one, as OpenScrub does) *lost* 42 detections
because a face three-quarters visible gets a narrower mirror-centred box that wins the NMS; the
shipped design is a union (normal pass unchanged, padded pass only adds non-overlapping boxes):
0 lost, 56 gained. By eye on the contact sheet about a quarter of the gains are real faces cut
off at the top of the picture (people close to the camera); the rest are hands, backs of heads,
tyres and, repeatedly, the Axon timestamp overlay reflected into a symmetric blob at 0.7-0.8
confidence, which under the precise profile would become a persistent fake face track over the
very overlay that must stay readable. Cost 1.6x face detection. Decision: shipped as an option
in the Auto-detect dialog, **off by default**; the anchored tracker already follows a face to
the border once it has been boxed, which is the body-cam case that matters.

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
