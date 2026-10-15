# Phase 3 (speed) - measurements and decisions (2026-10-15)

Rule from Phase 1: nothing ships without the benchmark; frame skipping only with
tracking coverage and no rise in leak rate. Baseline: bench/results/baseline.json.

## 1. Decode paths (bench/bench_decode.py, 600 frames)
| path | 720p fps | 1080p fps |
|---|---|---|
| OpenCV VideoCapture, full res | **764** | **264** |
| FFmpeg raw pipe, CPU, full res | 161 | 77 |
| FFmpeg pipe, CPU, proxy 640 | 499 | 405 |
| FFmpeg pipe, QuickSync, proxy 640 | 479 | **507** |
| FFmpeg pipe, NVDEC (cuda), full / proxy | 53 / 289 | 64 / 285 |
| FFmpeg pipe, d3d11va, full / proxy | 201 / 302 | 74 / 196 |

Full-resolution raw pipes are bandwidth-bound (a 1080p BGR frame is 6 MB), so
hardware decoders cannot beat OpenCV for the export path. Where pipes win is
delivering a *downscaled proxy* directly: QuickSync at 640 px holds ~500 fps at any
source resolution. Decision: `core/streams.FrameProducer` uses OpenCV for full
resolution and an FFmpeg pipe (QuickSync -> CPU fallback) for proxies; auto-detect
asks for a 1280 px proxy when the source is wider (1080p/4K body cams, dash cams).
PyAV was not needed. NVENC/QSV/AMF/MF *encoders* were measured earlier at 400-500 fps,
same as libx264 at 720p: encode is not the bottleneck, so no hardware encode.

## 2. Inference engines (bench/bench_providers.py; YuNet via the dynamic-dims ONNX)
| engine | YOLOX-tiny 416 | YOLOX-s 640 | YuNet 1280x736 |
|---|---|---|---|
| ONNX Runtime CPU (4 threads) | 17 ms | 83 ms | 10 ms (model only) |
| OpenCV DNN CPU (previous YuNet path) | - | - | 26-29 ms |
| OpenVINO API, CPU | 32 ms | 115 ms | 13 ms |
| **OpenVINO API, Intel UHD 770 iGPU (FP16)** | **11 ms** | **32 ms** | **8 ms** |
| OpenVINO API, RTX 3050 Ti (via OpenCL) | 43 ms | 127 ms | 12 ms |
| ONNX Runtime CUDA / TensorRT | not loadable: ORT-GPU 1.31 needs CUDA 13 runtime DLLs that pip does not provide; 1.22 + cu12 wheels still fail on cuDNN | | |
| ONNX Runtime OpenVINO EP | not loadable: EP wheel is ABI-bound to one OpenVINO release; every published runtime (2025.4-2026.4) fails to load | | |

Decisions:
- **YuNet moved from OpenCV DNN to ONNX Runtime** (`detectors/yunet_ort.py`): identical
  detections on 99 frames, 15.6 ms vs 21.9 ms end to end (model 10 ms + blob/NMS).
- **Intel tier = OpenVINO Python API** (`vision/runner.py`), not the ORT EP: YOLOX-tiny
  11 ms, YuNet 8 ms on the iGPU and the CPU stays free for decode/tracking. Compiled
  models are cached on disk (first compile 5-12 s). Falls back to ONNX Runtime CPU on any
  failure. The `openvino` wheel (Apache-2.0, ~100 MB) becomes part of the installer.
- **NVIDIA tier parked**: the fleet has no NVIDIA GPUs; `accel.py` still detects it and
  the runner will use the CUDA/TensorRT providers if a working `onnxruntime-gpu` is
  present (engine cache paths are wired), but no pip-installable combination worked here.
- Batched inference: YOLOX ONNX accepts a dynamic batch after a dim patch (CPU gain only
  19.9 -> 17.6 ms/frame at batch 8); OpenVINO/iGPU batching is the next step once the
  Intel tier is in the installer. `Detector.detect(frames)` already takes batches.

## 3. Pipeline changes
- `core/streams.py`: `FrameProducer` (decode thread, bounded queue, OpenCV or FFmpeg
  proxy pipe with QuickSync) and `FFmpegWriter` (encode thread). Export is now
  decode -> redact -> encode with the three overlapping; encode+pipe time fell from
  15.6 s to 4.7 s on three clips.
- Compositing: ellipse / rounded masks use `cv2.copyTo` (numpy boolean indexing cost 39 s
  on three clips vs 1.1 s for rectangles).
- Auto-detect reads from the producer; **scene-cut detection** (HSV histogram distance
  > 0.55 between detection frames) forces a detection and resets carriers/tracker;
  **adaptive stride** stretches the detection interval up to 15 frames while the scene
  is static (distance < 0.02) and there are live tracks. On the crowd clip both are
  inert (coverage 100 % with and without), so the gate holds; gains come on static scenes.
- Carry-tracking cost cut: every 3rd frame, 2 s limit, at most 8 concurrent carriers.
- Multi-scale faces: the 1.5x pass runs on alternate detection steps.
- `core/batch.py` + File > Batch process: queue of files, detect + save project (+ export
  with verification), JSON summary; files run one after another because each file
  already saturates the cores with its own workers.

## 4. Results (same three 45 s clips as the baseline, single process, bench/run_bench.py)
| | auto-detect fps | export fps | detect s | carry s | redact s | encode+pipe s |
|---|---|---|---|---|---|---|
| Phase 1 baseline (old pipeline: overlap linking, single-scale YuNet via OpenCV, no head fallback) | 56 | 177 | 56.5 | 12.9 | 1.1 | 15.6 |
| Phase 2 accuracy pipeline, before Phase 3 | 30 | 83 | 76.7 | 51.0 | 39.6 | 4.7 |
| Phase 3, CPU tier | 42 | 182 | 61.6 | 23.1 | 11.6 | 3.6 |
| **Phase 3, Intel iGPU tier (OpenVINO)** | **63** | 180 | **27.2** | 25.5 | 11.7 | 3.7 |

So the accuracy pipeline from Phase 2 (multi-scale faces, head fallback, BoT-SORT, 25 % pads,
ellipse masks, 15-frame persistence) now runs *faster* than the original pipeline did, on the
same hardware, with no change to what it finds (coverage gate 100 %, detections identical
across engines). On the 3-minute crowd slice with parallel workers: Phase 2 end 60 fps ->
**111 fps** (old-style configuration 172 -> 248 fps).

Export: the decode/encode overlap and the compositing fix recovered the Phase 2 regression
(83 -> 180 fps); the remaining cost is compositing ~150 tracks per frame (2.9 ms). The machine's
encode capacity is the ceiling: two concurrent exports total 356 fps, and segment-parallel
export was *slower* (169 vs 182 fps on the full 14-min file), so it stays opt-in
(`ExportOptions.workers`). Sources on a network share export 40 % slower than local; they are
now staged to local disk automatically (`media.local_path`, cached, 6 GB cap).

Frame skipping: adaptive stride and scene cuts kept coverage at 100 % on the gate clip and are
inert on it (crowd scene never static); expected to pay off on static interview/vehicle scenes.
Gate rule stays: any future change to stride/carry must re-run `bench/eval_autodetect.py`.

Not done / parked: CUDA/TensorRT engines (no working pip runtime; no NVIDIA in the fleet),
INT8 quantisation (needs the labelled set to check recall), batched iGPU inference (next step),
PyAV (unnecessary: the FFmpeg pipe producer covers it).
