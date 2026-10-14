# Phase 2 (accuracy) - status, measurements, decisions (2026-10-14)

All numbers are from our footage (bench/frames: 495 frames from 11 clips of the two Axon
recordings) and are **provisional**: ground truth is not labelled yet, so "recall" below is
against a consensus of detectors (a face counts if >= 2 detectors found it, IoU >= 0.4), with
every detector's unique finds checked by eye on contact sheets (bench/results/faces2/*.jpg).

## 1. Pluggable detector interface - done
`videoredact/vision/detectors/`: `Detector.detect(frames) -> list[list[Detection]]`,
adapters for YuNet, CenterFace, SCRFD, YOLOX, EgoBlur, Grounding DINO; combinators
`TiledDetector` (SAHI-style), `MultiScaleDetector`, `EnsembleDetector` (class-aware
NMS + weighted box fusion), `HeadFallback`; `build(cfg)` assembles a pipeline from
config. Auto-detect builds its detector through it (`auto_detect.build_detector`).

## 2. Faces - evaluated, decided
| detector | recall vs consensus | unique finds (eyeballed) | ms/frame (720p, CPU) |
|---|---|---|---|
| YuNet @960 | 0.45* | 16 | 18 |
| **YuNet @1280** | **0.93** | 0 | 26 |
| YuNet @1280 + tiles 640 | 0.95 | 23: mostly steering-wheel logos, hands, keyboards | 101 |
| **YuNet @1280 + multi-scale x1.5** | **0.96** | 12: nearly all real small / distant / profile faces | 65 |
| CenterFace @1280 | 0.29* | 3 | 192 |
| SCRFD-10G @640 | 0.69 | 1 | 105 |
| SCRFD-10G @640 + tiles | 0.83 | 26 | 591 |
| EgoBlur face | not run: weights are only available via click-through at projectaria.com (see below) | | |
\* from the first run before the SCRFD decoding bug was fixed; the consensus then contained
SCRFD's sky-patch false positives, which depressed every recall. The second run (faces2) is authoritative.

Decision: **YuNet @1280 with multi-scale (1.0, 1.5) is the default face detector** (MIT, no
licence issue). SCRFD (research-only weights) adds nothing on this footage. Tiling stays as an
option for 4K sources. SCRFD-34G not evaluated: the only public pack ships det_10g.
EgoBlur: download `ego_blur_face.jit` and `ego_blur_lp.jit` from
https://www.projectaria.com/tools/egoblur/ into `%LOCALAPPDATA%\VideoRedact\models\eval\`
and run `bench/eval_faces.py --candidates yunet1280,yunet_ms,egoblur` with the eval venv
(`C:\dev\venv-eval`, has torch). The adapter is in `detectors/adapters.py`.

Also done: face/head boxes dilated 25 % by default (Settings > "Face/head margin"), rendered
as ellipses (rounded rectangle also available); head fallback emits a `head` box for every
detected person without a face (top 22 % of the person box, 45 % width; skipped for wide
lying boxes). A learned head class replaces the heuristic once the fine-tune in `train/` is run.

## 3. Screens, documents, IDs, plates - open-vocabulary
Candidates: YOLOE-26 (Ultralytics) is **AGPL** -> evaluated only if you accept relicensing or a
commercial licence; not shipped. RF-DETR is Apache but closed-vocabulary (COCO). The permissive
open-vocabulary option is **Grounding DINO tiny** (Apache, via transformers, CPU) - adapter
`detectors/openvocab.py` with the prompt list from the spec, user-editable
(`settings["open_vocab_prompts"]` planned once wired into the dialog). Measured timings are in
this file's "Open-vocabulary test" section below once the run completes. It is too slow for every
5th frame on CPU, so the intended use is: run every ~1-2 s for static objects (screens,
documents, plates) and let carry-tracking / BoT-SORT fill between, or the offline mode.
License plates: EgoBlur LP needs the manual download above; open-image-models YOLOv9-t (MIT)
is the lighter alternative (training data undisclosed - verify before shipping).
Fine-tuning: `train/finetune_yolox.py` + `train/README.md` (YOLOX, Apache; no Ultralytics).

## 4. Temporal robustness - done
- Multi-object tracking through Roboflow `trackers` (Apache-2.0): **BoT-SORT with camera-motion
  compensation (sparse optical flow)** is the default association; ByteTrack / OC-SORT / overlap
  selectable. Kalman prediction is fed the real timestamps so it spans the detection stride.
- Two-pass is the existing design (detect+track -> track database in the project -> render).
- Back-fill before the first detection and persistence after the last: `lead_in` / `lead_out`
  (default 15 frames, Settings > "Lead-in / persistence").
- Gap interpolation and fragment linking unchanged (carry-tracking with ViTTrack between detections).
- SAM 3 "high-accuracy offline mode": **not enabled**. The target fleet has no GPU; SAM-family
  video propagation on CPU is seconds per frame (research notes: EfficientViT-SAM/MobileSAM
  encoders 1-3 s per frame). Licence of SAM 3 (Meta "SAM License") also needs review. Plan:
  click-to-mask on a paused frame with EfficientViT-SAM-L0 ONNX, propagated by the tracker,
  as an optional pack; benchmark on the dev laptop's RTX 3050 Ti first.

## 5. Human review - improved
- Redactions panel has a **Review** column with flags: `short` (< 1 s), `single` (one
  detection), `low-conf` (< 0.6), `head-fallback`, `linked`, `incomplete`. "Next flagged"
  button; keys **F / Shift+F** cycle flagged tracks.
- Keys **[ / ]** trim the selected track's start/end to the playhead, **=** extends it to the
  playhead; manual boxes still auto-track forward/backward with re-acquisition.
- Timeline lanes per track and live-updating tracks unchanged.

## 5b. Whole-pipeline measurement (crash-scene clip, 3 min from 5:00, faces, parallel workers)
Reference = YuNet@1280 single-scale on every 15th frame (577 faces); "coverage" is how
many of those the produced tracks cover >= 60 %. It saturates by construction - the
informative columns are frame-boxes (how much gets redacted), tracks (review burden) and fps.

| configuration | fps | tracks | redacted frame-boxes | face coverage |
|---|---|---|---|---|
| v0.1.7 (overlap association, no head fallback, single scale) | 172 | 282 | 15,235 | 100 % |
| + BoT-SORT identity (additive) | 162 | 214 | 11,871 | 100 % |
| **new default**: BoT-SORT + head fallback + multi-scale + 25 % pad + lead 15 | 57 | 568 -> 464 with 2 s linking | 68,653 | 100 % |

Reading it: the new default redacts 4.5x more frame-boxes (turned-away heads, distant faces,
persistence around every detection) at a third of the speed, and 224 of its tracks are single
detections (kept on purpose, flagged `single` for review). The first BoT-SORT wiring dropped
coverage to 89 % because the library returns smoothed boxes and silently drops unconfirmed
detections; the fix keeps raw detection boxes and falls back to overlap matching for anything
the tracker drops, which is why it is now "additive".

## Open-vocabulary test (Grounding DINO tiny, CPU, 800 px)
Load 29 s; **8.8-9.8 s per frame**; detections on the test frames were low-confidence
(0.30-0.44) and mostly wrong (plates/ID cards on frames without any). Not viable on the
target fleet even at a sparse rate. Decision: skip open-vocabulary at runtime; get screens,
documents, IDs and plates from the fine-tuned YOLOX (`train/`) once labels exist. The adapter
stays in the repo for offline experiments (eval venv only).

## 6. Still open from the Phase 2 spec
- Wire the open-vocabulary detector + prompt editor into the auto-detect dialog (after timing).
- Learned head/plate/ID classes via `train/` once labels exist.
- SAM-based masks (see above).
- Ground-truth benchmark run (`bench/run_bench.py`) after labeling to replace the consensus numbers.
