# Technical upgrade options (permissive licenses only), Oct 2026

Condensed research notes plus our own measurements. **[U]** = unverified.
Rule: MIT/Apache/BSD code and weights only; no GPL/AGPL; watch training-data licenses.

## Head / face detection
- SCRFD: code MIT but pretrained weights non-commercial -> excluded.
- YuNet (current, opencv_zoo MIT) handles small and side faces; trained on WIDER FACE (CC BY-NC-ND) -> document this in licensing notes.
- Head detectors trained on CrowdHuman/SCUT-HEAD inherit non-commercial dataset terms; most are YOLOv5/v8 (AGPL) anyway. PINTO YOLOv9-Wholebody head models are GPLv3 -> excluded.
- Permissive path for backs of heads/helmets: fine-tune an Apache detector (our YOLOX, RT-DETR, D-FINE, RF-DETR) on CC BY 4.0 head datasets from Roboflow Universe, adding a "head" class to YOLOX-tiny at zero extra runtime cost.

## License plates
- open-image-models (ankandrew), MIT, YOLOv9-t/s plate detectors 256-640 px, mAP50 0.86-0.97, ONNX. Training data license undisclosed **[U]**. Fine-tune data: Roboflow "License Plates" (CC BY 4.0), CC0 Kaggle set. Avoid OpenALPR (AGPL).

## Multi-object tracking ("redact every face")
- ByteTrack (MIT), OC-SORT (MIT); roboflow/supervision ships an MIT ByteTrack to copy. boxmot is AGPL: never vendor.
- Essentials: Kalman on (cx, cy, aspect, h); two-pass association (confident first); long track buffer (30-60 frames); offline gap interpolation; dilate boxes 10-20 %.
- ReID: LibreReID OSNet-AIN x0_25 (MIT, 1 MB, 512-d) for re-identification across gaps **[U weights provenance]**; SFace (OpenCV) for per-person review cards.
- Detect-every-N + propagate between detections is what redactcam/OpenScrub do. **Implemented in v0.1.5**: detections every N frames, ViTTrack carry through misses, parallel chunk workers.

## Segmentation masks on CPU
- MobileSAM (Apache), EfficientViT-SAM (Apache, official ONNX), SAM 2.1 tiny (Apache, community ONNX). Realistic for click-to-mask on a paused frame (encoder 1-3 s, then ms per click), not per frame. Avoid EdgeSAM (non-commercial), FastSAM (AGPL).

## Audio
- Diarization: sherpa-onnx (Apache) = pyannote segmentation-3.0 (MIT) + 3D-Speaker / WeSpeaker embeddings (Apache); CPU real-time factor ~0.1 reported.
- Keyword spotting: sherpa-onnx KWS (Apache). openWakeWord models are CC BY-NC-SA -> avoid.
- Denoising: RNNoise BSD-3; GTCRN Apache; DeepFilterNet weights license unclear and it *worsened* whisper WER in one report. Recommend loudnorm + optional denoise A/B only.
- Whisper large-v3-turbo int8: better on noisy audio, likely 3-6x real time on 8 cores **[U]**; offer as "accurate" mode.
- Forced alignment: wav2vec2-base-960h (Apache) with torchaudio CTC; MMS_FA is CC BY-NC -> avoid.

## Acceleration without a GPU
- ONNX Runtime OpenVINO EP (MIT/Apache): 30-60 % faster on Intel CPUs, up to ~4x with INT8; can use Intel iGPU. Biggest cheap win for typical government PCs **[U numbers vendor-adjacent]**.
- DirectML EP: maintenance mode; fallback for AMD iGPUs.
- FFmpeg hardware decode: inconsistent gains; decode is not our bottleneck (777 fps at 720p measured).
- **Our measurements (i7-14700T, 720p body-cam):** ORT thread spinning off + 4 intra-op threads = 2.5-5x speedup of everything else; KCF 3.4 ms, ViTTrack 5 ms (scale-adaptive), CSRT 45-72 ms; YuNet@1280 29 ms; YOLOX-tiny 26 ms; hardware encoders no faster than libx264 at 720p (all ~400-500 fps).

## Open-source UIs worth studying
- OpenScrub (Apache): per-person review cards (SFace), stackable detection windows, gap bridging, safety bands for un-OCR'd text.
- redactcam (MIT code): sparse detection + LK propagation, tiled detection, **second independent re-detection pass that fails the export if a mask is missing** (adopted as our post-export verification).
- open-redactor (Apache): coverage-gap reports instead of silent concealment.

## Prioritized upgrades
| # | Upgrade | Impact | Effort | Status |
|---|---|---|---|---|
| 1 | Multi-track with carry between detections, gap interpolation, dilation | every face tracked, fewer gaps | M | done v0.1.5 |
| 2 | OpenVINO EP (INT8) for detectors | 1.3-4x detector speed | S | todo |
| 3 | "head" class via fine-tune on CC BY head data | backs of heads, helmets | M | todo |
| 4 | Plate detector: open-image-models YOLOv9-t | new capability | S | todo (check data license) |
| 5 | Verification re-detection pass before release + coverage report | defensibility | M | done v0.1.5 |
| 6 | whisper large-v3-turbo option + loudnorm | noisy-audio accuracy | S | todo |
| 7 | sherpa-onnx diarization (mute one speaker) | per-speaker redaction | M | todo |
| 8 | wav2vec2 forced alignment | tighter word bleeps | S | todo |
| 9 | Click-to-mask (EfficientViT-SAM) on paused frame | silhouette masks | M | todo |
| 10 | OSNet/SFace per-person review cards + re-ID | review speed | M | todo |
| 11 | sherpa-onnx keyword spotting | fast PII audio review | S | todo |
| 12 | Hardware decode flag | frees CPU | S | low value |
