# VideoRedact: current architecture and weaknesses (Phase 0 audit, 2026-10-14, v0.1.7)

Line numbers refer to the commit tagged v0.1.7 plus the Phase 0 `accel.py` change.

## 1. Stack
- **Language / framework:** Python 3.11+ (dev venv is 3.13), PySide6 (Qt 6.11) desktop UI, PyInstaller one-folder build, Inno Setup installer, GitHub Releases update notification, notify-only (`videoredact/updater.py`).
- **Vision runtime:** OpenCV 5.0 (contrib, CPU build with OpenCL/NVD3D11 only, no CUDA; `cv2.getBuildInformation`), ONNX Runtime 1.30 CPU wheel (`onnxruntime`), `onnx` for one graph rewrite.
- **Speech:** faster-whisper 1.x on CTranslate2 (int8, CPU).
- **Media:** FFmpeg 7.1 binary (imageio-ffmpeg wheel in dev; `bin/ffmpeg.exe` in the installer, GPL "essentials" build, LGPL swap still pending) for probe/decode/encode; OpenCV `VideoCapture` (FFmpeg backend) for frame access.
- **Models (all MIT/Apache):** YuNet face detector (`face_detection_yunet_2023mar.onnx`, via `cv2.FaceDetectorYN`), CenterFace (optional, ONNX Runtime), YOLOX-tiny/-s COCO detectors (ONNX Runtime), ViTTrack single-object tracker (`object_tracking_vittrack_2023sep.onnx`, via `cv2.TrackerVit`), whisper `small` int8. Registry and downloader: `videoredact/vision/models.py:30-50`, `ensure_model` at `models.py:68`.

## 2. Pipeline as built

### 2.1 Media access (`videoredact/core/media.py`)
- `probe()` (`media.py:84`) parses FFmpeg's stderr banner with regexes (`media.py:79-81`) and then opens the file with OpenCV for exact frame count/fps.
- `decode_audio()` (`media.py:138`) spawns FFmpeg to pipe f32le PCM (whole file or a range) into memory.
- `FrameReader` (`media.py:166`) wraps `cv2.VideoCapture`; `seek()` (`media.py:185`) grabs forward for hops <= 30 frames (`media.py:191`) and otherwise sets `CAP_PROP_POS_FRAMES` (`media.py:197`), which is keyframe-seek-plus-decode and can be hundreds of ms on long GOPs.

### 2.2 Detection (`videoredact/vision/detector.py`)
- `FaceDetector` (`detector.py:58`): YuNet through OpenCV DNN; frame downscaled to `max_side` (`detector.py:67-68`), 1280 when used by `CombinedDetector` (`detector.py:238`), 960 default otherwise.
- `CenterFaceDetector` (`detector.py:101`): dynamic-shape rewrite of the published model, deface preprocessing; opt-in only (slower, fewer hits on our footage).
- `ObjectDetector` (`detector.py:160`): YOLOX ONNX, letterbox to 416/640, per-class NMS, COCO ids mapped to privacy labels (`detector.py:23`).
- `CombinedDetector` (`detector.py:229`) runs face + object detectors sequentially on the same frame.
- ORT sessions come from `_ort_session()` which now takes the provider chain from `vision/accel.py` (`accel.py:78`, `accel.py:152`; session factory `detector.py:83-99`) with CPU fallback; thread budget 4 intra-op, spin-waiting disabled (`detector.py:90`).

### 2.3 Single-object tracking, draw-a-box (`videoredact/vision/tracker.py`)
- `ObjectTracker.track()` (`tracker.py:391`): adaptive work frame so the object is ~110 px (`tracker.py:50`, `_adapt_work_side` `tracker.py:192`), ViTTrack default (`TrackOptions.tracker`), every 2nd frame processed and interpolated, periodic detector snap for known classes, HSV-histogram drift check (`tracker.py:43-44`), loss after 3 bad frames (`tracker.py:42`), lost-mode search = multi-scale template match over the whole work frame (`tracker.py:47`, `_search` `tracker.py:245`) plus class detector; backward pass reads 64-frame chunks in reverse (`_backward` `tracker.py:291-294`), stops after 4 s lost.
- Live snapshots to the UI every 45 processed frames (`tracker.py:39`).

### 2.4 Automatic detection ("redact every face") (`videoredact/vision/auto_detect.py`)
- `run_auto_detect()` (`auto_detect.py:360`): splits the frame range into up to 4 chunks for `multiprocessing` spawn workers (`auto_detect.py:369-387`) when the range is >= 1500 frames (`auto_detect.py:370`); each worker runs `_detect_chunk()` (`auto_detect.py:157`): detectors every `stride` (5) frames on the full-resolution frame, two-pass greedy IoU association (`_match` `auto_detect.py:93`), ViTTrack *carry* of tracks that miss a detection (`_Carrier` `auto_detect.py:124`, constants `auto_detect.py:43-52`), single detections kept (`auto_detect.py:51,64`), fragment linking (`link_fragments` `auto_detect.py:279`), chunk merge by IoU (`_merge_chunks` `auto_detect.py:330`).

### 2.5 Redaction rendering (`videoredact/core/video_redact.py`, `audio_redact.py`)
- `apply_box()` (`video_redact.py:43`): black fill, pixelate with 8 blocks across the short side (`video_redact.py:17,27-30`), or downscale-blur-upscale (`video_redact.py:32-38`); rectangle or ellipse mask; padding fraction per track.
- `redact_frame()` (`video_redact.py:62`) iterates enabled tracks and their interpolated box at the frame; used identically by the preview player and the exporter.
- Audio: `apply_redactions()` replaces ranges with beep/silence/tone/noise with raised-cosine fades *outside* the range (`audio_redact.py`).

### 2.6 Export (`videoredact/core/export.py`)
- `export_project()` (`export.py:58`): decode audio -> redact -> temp WAV; then sequential `FrameReader.iter_frames` (`export.py:137`), `redact_frame` per frame in Python (`export.py:139`), raw BGR piped to FFmpeg libx264 (`export.py:125,140`); SHA-256 of output and source (`export.py:171-174`); optional clip range; optional post-export verification (`export.py:192-193`, `core/verify.py:62`) re-running the face detector on the output every 15 frames and reporting detections < 60 % covered (`verify.py:63`).

### 2.7 Speech (`videoredact/audio/transcribe.py`, `pii.py`)
- `Transcriber` (`transcribe.py:38`): faster-whisper, int8, VAD off by default, beam 5, word timestamps, filler/no-speech filter (`transcribe.py:23-24,73`), segments streamed to the UI; offline load when the model exists locally (`transcribe.py:47`).
- PII suggestions are regex/cue heuristics only (`pii.py:18-27`, `suggest_pii` `pii.py:57`).

### 2.8 UI and persistence
- Own preview player (`videoredact/ui/player.py:95`): OpenCV frames + FFmpeg PCM pipe into `QAudioSink`, audio clock (`player.py:165`), 8 ms tick (`player.py:41`), frame skipping when behind (`player.py:278-281`).
- Jobs dock (`ui/jobs.py`) runs transcribe/track/detect/export/hash in QThreads with partial-result streaming.
- Project = JSON `.vrproj` (`core/model.py`): transcript, audio redactions, video tracks as per-span keyframes (normalized boxes), audit log; report PDF/CSV (`core/report.py`).

## 3. Hardware this runs on (detected 2026-10-14 on the dev laptop)
- Windows 11 Pro 10.0.26200; Intel i7-14700T (20 cores / 28 threads), 64 GB RAM.
- NVIDIA GeForce RTX 3050 Ti Laptop (4 GB), driver 595.95, CUDA 13.2 capable.
- Intel UHD Graphics 770 iGPU. No NPU.
- Installed runtimes: ONNX Runtime CPU only; OpenCV without CUDA (OpenCL via D3D11 present). Result: `tier=cpu`, with notes that both NVIDIA and Intel accelerated wheels are absent. Target agency machines are expected to be 8-core CPUs with Intel iGPUs and no NVIDIA GPU.

## 4. Acceleration tiers (design, Phase 0 implemented detection + selection)
`videoredact/vision/accel.py`:
- `detect_hardware()` gathers CPU/RAM/adapters (CIM), `nvidia-smi` (name, driver, VRAM, CUDA), NPU PnP names, and `onnxruntime.get_available_providers()`.
- `_choose_tier()` (`accel.py:134`) picks the highest tier whose runtime is actually installed: **nvidia** (TensorRT EP, else CUDA EP) -> **intel** (OpenVINO EP, device NPU if present else GPU) -> **directml** (DmlExecutionProvider, other DX12 GPUs) -> **cpu**. Missing runtimes are reported as notes (shown by `--selftest`).
- `preferred_providers()` returns the EP chain with CPU last; `detector._ort_session()` tries it and falls back to CPU on any session error, so a bad driver never breaks detection.
- Not yet done (Phase 1+): bundling `onnxruntime-openvino` / `onnxruntime-gpu` wheels (they are mutually exclusive packages; the installer needs per-tier variants or a runtime-selected side-by-side layout), moving YuNet/ViTTrack off OpenCV DNN onto ORT so they benefit from the tier, FP16/INT8 model variants, and FFmpeg hardware decode (`-hwaccel d3d11va/qsv`, measured gains inconsistent).

## 5. Weaknesses found

### Accuracy
1. **Faces only when frontal-ish.** YuNet (`detector.py:58`) misses backs of heads, helmets, strong profiles; no head detector. Auto-detect coverage is therefore bounded by what YuNet sees; the verification pass (`verify.py:62`) uses the *same* detector, so it cannot catch what the detector never sees.
2. **No license-plate detector.** COCO has no plate class (`detector.py:23`); plates rely on draw-and-track.
3. **"document" = COCO "book"** (`detector.py:23`); notepads, forms and ID cards are detected inconsistently. "screen" covers laptops/TVs but not phone screens (mapped to "phone").
4. **IoU-only association** (`auto_detect.py:93`) has no motion model; fast pans or crossing people can swap identities between detection steps (stride 5 = 167 ms at 30 fps). Fragment linking (`auto_detect.py:279`) uses heuristics (IoU >= 0.2 or centre distance), not appearance.
5. **Carry is blind to identity**: `_Carrier` (`auto_detect.py:124`) keeps following whatever the tracker locks onto; the HSV histogram check (`auto_detect.py:46`) is the only guard, weak on similar clothing/backgrounds.
6. **Single-object re-acquisition** after loss is template matching on the original patch (`tracker.py:245-266`, scales `tracker.py:47`): fails on rotation/lighting change; no appearance embedding (ReID).
7. **Fixed thresholds everywhere** (face 0.5, object 0.35, IoU 0.3, histogram 0.60/0.38, template 0.60; `tracker.py:41-53`, `auto_detect.py:43-52`) with no per-video adaptation and no calibration set; coverage numbers come from one 3-minute clip.
8. **Pixelate/blur strength** is relative to the box (`video_redact.py:17-18`); large boxes get coarse blocks, but a small face box at 8 blocks can be re-identifiable; research says only solid fill is safe, yet blur/pixelate are offered without warning.
9. **PII suggestions are regex cues** (`pii.py:18-27`); spoken names not preceded by a cue, spelled-out addresses and most dates are missed.
10. **Whisper `small`** (`transcribe.py:40`) on noisy body-cam audio: usable but error-prone on names; no loudness normalisation, no denoise, no diarization.

### Speed
11. **Everything per-frame is Python.** Export (`export.py:137-140`) decodes, redacts and pipes every frame through Python (~150-170 fps at 720p); a 47-min 720p file takes ~8.5 min; 1080p would be ~2.5x slower. No zero-copy, no GPU, no batching.
12. **Detectors run one frame at a time** (`detector.py:65,199`) with no batching, and two separate ORT/OpenCV-DNN inferences per detection frame (`detector.py:242`). YuNet runs through OpenCV DNN (CPU only, no provider tiering); ViTTrack likewise (`tracker.py` `make_tracker`).
13. **Sequential decode is duplicated**: auto-detect workers each decode their chunk (`auto_detect.py:312`), the single-object tracker decodes again, verification decodes the output again, export decodes again. No frame cache, no proxy (low-res) file.
14. **Backward tracking** reads chunks in reverse via seeks (`tracker.py:291-294`): each 64-frame chunk is a keyframe seek plus decode; cost grows with GOP length.
15. **Parallel detection scales poorly** when the single process already uses all cores (ORT 4 threads + OpenCV threads): measured 1.4-1.7x on 4 workers (`auto_detect.py:369`). Worker start-up (spawn + imports + model load) is 3-5 s each.
16. **ViT carry cost is linear in the number of tracks** (5 ms per track per processed frame; `auto_detect.py:48-49`): crowded scenes drop to ~85 fps single-process.
17. **Preview player** converts every frame to a QPixmap on the GUI thread (`ui/video_view.py`) and runs `redact_frame` on it; 1080p60 preview will drop frames (`player.py:278-281` skips ahead but display rate suffers).
18. **FFmpeg probe by stderr regex** (`media.py:79-81`) is brittle (variable frame rate, odd layouts, multiple streams); no `ffprobe` JSON.
19. **`FrameReader.seek`** (`media.py:185-197`) is slow for random access > 30 frames; UI scrubbing on long-GOP files feels sluggish.
20. **Model loading is repeated** per job (`main_window.py:687`, `auto_detect.py:312`): YuNet/YOLOX/ViT sessions are recreated for every tracking or detection run instead of being cached per process.

### Robustness / operations
21. **Settings and labels are duplicated in the UI** (`main_window.py:663` hard-codes the detector label set; `dialogs.py` repeats choices).
22. **No automated accuracy regression**: tests cover unit behaviour and smoke flows; the only quality metric is the ad-hoc benchmark on one clip (NOTES performance table). Needs a small labelled set and a `scripts/bench_quality.py` that reports coverage, fragments and fps per release.
23. **Frozen build is CPU-only** by construction (no accelerated ORT wheels bundled); `accel.py` will report the tier but cannot use it until Phase 1 packaging.
24. **Installer ships a GPL FFmpeg build** (NOTES) and the Restart Manager hang workaround exists only in the updater path.

## 6. Priorities implied by this audit (feed into Phase 1+)
1. Accelerated ORT tiers in the build (OpenVINO for the agency fleet; CUDA for dev/benchmark) and move YuNet/ViT to ORT so they benefit.
2. Head detector class (fine-tune YOLOX-tiny on CC BY head data) and a plate detector (open-image-models) to fix the biggest coverage holes.
3. Motion model (Kalman) + appearance embedding in association and carry; cache model sessions per process.
4. Shared low-resolution proxy decode for detection/tracking/verification; batch inference where the EP supports it.
5. Labelled quality benchmark run on every release.
