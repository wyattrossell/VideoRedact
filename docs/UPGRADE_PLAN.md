# Upgrade plan (proposed after Phase 0 audit and Phase 1 benchmark) - awaiting approval

Baseline numbers: `bench/results/baseline.json` (2026-10-14, 11 x 45 s 720p30 clips cut
from the two Axon samples, 14,828 frames, dev laptop at the CPU tier; accuracy pending
your labels, see bench/README.md). Weaknesses referenced as W1..W24 are in
docs/ARCHITECTURE_CURRENT.md section 5.

Where the time goes today (aggregate of the 11 clips, single process):
- Auto-detect (face+screen+document+phone, stride 5): **60 fps**. Detection = 86 % of
  the time (72 ms per detection call = YuNet@1280 + YOLOX-tiny), carry-tracking 9 %,
  decode 5 %, association < 1 %. The 45 s clips are below the 1500-frame threshold at
  which the parallel path engages, so the parallel column equals single-process; on a
  3-minute slice the parallel path measured ~140 fps for faces only (NOTES).
- Export: **202 fps**. Encode + pipe 66 %, decode 29 %, redaction compositing 5 %.
Conclusion: detector inference is the lever for auto-detect; encode/pipe overlap and
decode for export.

## Goals and how we will know
| Goal | Metric (bench/run_bench.py) | Baseline (CPU, dev laptop) | Target |
|---|---|---|---|
| Fewer leaks | `leak_rate` on labelled frames | to be measured after labeling | < 2 % of frames, 0 for faces in frontal view |
| Faces/heads from any angle | `recall@90` face | tbd | > 95 % |
| Plates and ID cards | `recall@90` plate, id_card | 0 (no detector) | > 90 % |
| Auto-detect speed | `autodetect_fps` (single) / `autodetect_parallel_fps` | 60 fps / ~140 fps (long clips, faces only) | > 250 fps single CPU-tier, > 1000 fps GPU-tier |
| Export speed | `export_fps` | 202 fps @720p | > 400 fps |
| Review burden | tracks per minute of footage | 47 (per 45 s clip avg, singles kept) | < 15 per minute with the same leak rate |

## Architecture changes (in order)

### A. Shared decode + frame proxy (W11, W13, W19)
One decode pass per job producing (a) full-res frames for export and (b) a low-res
proxy (<= 640 px) for detection/tracking, via FFmpeg with a `scale` filter piped as
raw frames (measured 313 fps CPU at 512 px, decode no longer re-done per stage).
Verification and the single-object tracker read the proxy. Expected gain: removes
2 of 4 decode passes and all per-stage resizes; ~1.3x on auto-detect, ~1.2x on export.
Risk: low. Memory: proxy ring buffers only.

### B. Accelerated inference tiers (W12, W23) - the big speed lever
- Move YuNet and ViTTrack from OpenCV DNN to ONNX Runtime so they use the tier chosen
  by `vision/accel.py` (they are ONNX files already; YuNet post-processing and the
  ViT template/search preprocessing are ~80 lines each).
- Intel tier: ship `onnxruntime-openvino`; FP16 on iGPU, INT8 models for CPU. Expected
  1.5-3x on YOLOX/YuNet on agency PCs (vendor numbers, to be confirmed on the bench).
- NVIDIA tier: ship `onnxruntime-gpu` (CUDA 12 build) as an optional "GPU pack"
  installer variant; TensorRT EP where the driver allows. Expected 5-10x on detection.
- Batching: detectors accept a batch of stride frames (4-8) per call; helps GPU tiers most.
Packaging: the three ORT wheels are mutually exclusive, so the installer ships
`onnxruntime` (CPU) plus optional packs placed in `_internal/accel/<tier>/` and chosen at
startup by `accel.py` (sys.path insertion before `import onnxruntime`). Risk: medium
(driver/DLL mismatches); mitigated by the CPU fallback already in `_ort_session`.

### C. Detection coverage (W1, W2, W3) - the big accuracy lever
- **Head detector**: fine-tune YOLOX-tiny (Apache) on CC BY 4.0 head datasets (Roboflow
  "human head detection" sets, ~8k images) plus ~500 frames from our own labelled set,
  adding classes `head` and (from open data) `plate`, `id_card`, `document`. One model,
  one inference, replaces the COCO mapping. Expected: faces from behind/side and
  helmets covered; plates detected (currently 0).
- **License plates**: start with open-image-models YOLOv9-t (MIT) as a second model
  while the fine-tune is trained; drop it once the fine-tuned YOLOX covers plates.
- Keep YuNet for frontal precision; union with head boxes (a face inside a head box
  is one object).
Risk: training needs a GPU box for a few hours (the dev laptop's RTX 3050 Ti suffices
for YOLOX-tiny); dataset license review for plates (training data of
open-image-models undisclosed - we will train on CC BY / CC0 data ourselves).

### D. Association and carry with identity (W4, W5, W6)
- Kalman constant-velocity motion model per track (ByteTrack-style, MIT re-implementation)
  so matching uses predicted position, not last position; two-pass matching stays.
- Appearance embedding for faces/heads (SFace from OpenCV zoo, permissive) stored per
  track; used to (1) reject carry drift, (2) re-link fragments across gaps up to 10 s,
  (3) cluster tracks into "persons" for the review gallery.
Expected: identity swaps on crossings eliminated in the bench clips; fragments per
minute down 3x; same leak rate. Risk: medium; adds ~3 ms per new track.

### E. Export path (W11)
- Redaction compositing in C-speed: pre-rasterize masks per frame with OpenCV
  (`fillPoly`/`rectangle` + `bitwise_and`) and apply blur/pixelate only inside the
  mask union; frames handed to FFmpeg via a dedicated writer thread with a queue so
  decode, redact and encode overlap. Expected 2-3x export fps on 8 cores.
- Hardware decode (`-hwaccel d3d11va/qsv`) behind a flag, measured per machine.

### F. Review tooling (research items 1-3; required to realise the recall gains)
Review queue with thumbnails, approve/delete, "next unreviewed" hotkeys, and the
person gallery from D. Not a speed item but without it the extra recall becomes
reviewer burden.

## Sequence
1. Labels from you (20-50 clips) -> `baseline.json` with accuracy filled in.
2. A + E (pure engineering, no model changes) -> re-bench. Expect speed 2x.
3. B (OpenVINO first, as the fleet is Intel) -> re-bench on an Intel-only box too.
4. C (head/plate/ID model) -> re-bench accuracy; the leak rate target is judged here.
5. D -> re-bench identity swaps/fragments.
6. F in parallel with 4-5.

## Risks
- **Over-claiming**: every step re-runs `bench/run_bench.py` and commits the JSON; no
  change ships without a leak-rate that is equal or better on the same labelled set.
- **Model licensing**: all training data CC BY/CC0, code Apache/MIT; YuNet/WIDER FACE
  non-commercial caveat documented; no AGPL tooling (no Ultralytics for training - use
  the YOLOX repo's own trainer).
- **Packaging size/complexity** of accelerated runtimes; mitigated by optional packs and
  the CPU fallback.
- **Time**: C is the only step with real schedule risk (data prep + training + eval).

Stopping here for approval before changing the pipeline.
