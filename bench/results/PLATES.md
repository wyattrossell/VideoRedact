# Licence-plate detector: survey and first evaluation (2026-10-29, Phase 2 of the tracking rebuild)

Constraint: code, weights and training-data terms all permissive (MIT / Apache-2.0 / BSD); CPU,
ONNX Runtime or OpenVINO; no PyTorch in the installer. Web survey done by a research agent on
2026-10-29 (nothing downloaded there); the two candidates below were then downloaded and run
here on the 12 bench clips (`bench/eval_plates.py`, every 15th frame, 1 080 frames).

## Candidates

| Candidate | Licence status | Format | Verdict |
|---|---|---|---|
| **Topurrra `plate_rtdetr.onnx`** (RT-DETRv2-R18 fine-tuned on Open Images "Vehicle registration plate", 3 000 images) | Apache-2.0 code and weights; data CC-BY (Open Images) - the only candidate with clean end-to-end provenance | fixed 1x3x640x640, `logits` (1,300,1) + `pred_boxes` (1,300,4); scores compressed (top box ~0.1) | **Not shippable as-is**: 850 ms/frame on the CPU here (60 GFLOPs) and the detections on our clips are mostly not plates (see below) |
| open-image-models `yolo-v9-t-384-license-plates-end2end.onnx` | MIT repo, but trained with a GPL-3.0 YOLOv9 fork on a mix that includes an AGPL-3.0 repo (openalpr benchmarks) and web-scraped images; the author declines to warrant the weights | 1x3x384x384 letterboxed, (N,7) with NMS inside | **Benchmark / teacher only** (licence); 113 ms/frame with the CPU shared, 0.3 detections per frame |
| Meta EgoBlur LP (Faster R-CNN, TorchScript) | Apache-2.0 with click-through download; Meta-owned 200k-image training set | TorchScript, needs torch; ~1.5 s per 2 MP frame on CPU | Teacher for pseudo-labelling only |
| OpenCV Zoo LPD-YuNet | Apache-2.0 | ONNX 320x240 | Chinese plates; not for US footage |
| Ultralytics-based plate models (YOLOv5/8/11) | AGPL-3.0 | - | Disqualified |
| autolane rfdetr-alpr, justjuu rtdetr-v2, Smolry, platedet | unknown or missing data terms / no licence | - | Disqualified until provenance is known |

## What the two ONNX candidates found on our clips

`bench/eval_plates.py --models rtdetr=... yolov9t=... --stride 15` (scratch output, contact sheets
not committed):

| | frames | detections | ms / frame (CPU shared with another job) |
|---|---|---|---|
| rtdetr (thr 0.05) | 1 080 | 2 186 | 852 |
| yolov9t (thr 0.30) | 1 080 | 318 | 114 |

Contact sheets of the highest-scoring crops: both detectors fire mostly on the in-car laptop
screen and keyboard, a dealer "USA" plate frame and rear badge, "9/11/2001" shirts, notepads,
driver's licences and a "SERVICE" sign. The clips (a crash scene and a traffic stop) contain few
readable plates, so recall cannot be judged here, but precision on plate-like rectangles is poor
for both, and the RT-DETR speed rules it out for the sampled-frame pipeline anyway.

## Decision (for Saturday)

* Nothing ships for plates in the next release; plates stay a manual box (classic engine) until
  a detector earns its place. The adapter `vision/detectors/plates.py` is written for the RT-DETR
  format but not wired into the pipeline or the model registry.
* The clean path is our own detector: a permissive architecture (RT-DETRv2-R18 / D-FINE / RF-DETR
  Nano, all Apache-2.0, or YOLOX-tiny as already set up in `train/`) trained on Open Images
  "Vehicle registration plate" (CC-BY) plus our own body-cam frames pseudo-labelled by EgoBlur
  (Apache-2.0, teacher only) and reviewed locally, exported to ONNX at build time. Needs clips
  with plates in them (traffic stops at the car's rear) for both training and the gate.
* Gate when a candidate exists: contact sheet precision by eye on >= 3 clips with plates, then
  `bench/track_eval.py --classes license_plate` once auto-detect produces plate tracks.
