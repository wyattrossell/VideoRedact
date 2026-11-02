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

## Round two, 2026-11-02 pm: "plates still not working great, or at all at times"

Wyatt's verdict on the 0.1.19 build above. Three causes were visible in the pipeline without
any new footage:

1. **Scale.** The model sees a 384 px letterbox of the whole frame; a plate on a car across
   the street is a dozen pixels wide there and is simply not found.
2. **The gate loses the car before the plate.** Close up (an officer at the bumper) the object
   detector's vehicle box fades or is cut off while the plate is still fully readable, and the
   gate then drops a plate it had been confirming for seconds.
3. **Threshold.** 0.5 was chosen against the junk; behind a gate it costs plates at an angle.

What changed (`vision/detectors/plates.py` `VehiclePlates`, `vision/auto_detect.py`):

* **A crop of every detected vehicle is searched as well** (up to four per frame, largest
  first, box grown by 8 %, skipped when the vehicle fills more than 60 % of the frame). The
  plate in the crop is three to five times bigger than in the full-frame pass; the two sets
  are fused with the usual NMS.
* **Continuation without a vehicle box.** A plate the gate rejects is passed on marked
  `plate_ungated`; auto-detect keeps it only when it overlaps (IoU >= 0.3) a plate track the
  gate already confirmed twice, and it never starts a track.
* Plate threshold 0.5 -> 0.4 (PrivacyCam's value), plate track rule: 2 hits at a 0.4 average.
* The vehicle threshold was tried at 0.2 ("a wrong vehicle box costs nothing") and rejected:
  at 0.2 the patrol car's own interior counts as a vehicle and the in-car laptop, which the gate
  had removed entirely, came back (28 of 130 hits). It stays at 0.35.
* The gate is now an option in the Auto-detect dialog ("keep plates only where a vehicle is
  detected", on by default) for a clip where it is clearly wrong.

Measured with `bench/eval_plates.py --models pipeline=x --stride 15` on the same 1 080 frames
(the clips contain one real plate and one dealer plate, both on the stopped car; timings were
taken while other evaluations ran and are not comparable with the table above):

| configuration | hits | of which score >= 0.8 | what the sheet shows |
|---|---|---|---|
| 0.1.19 as shipped (thr 0.5, vehicle 0.35, no crops) | 22 | 6 | dealer plate 6 frames, the real plate once, paper on a hood 8, a licence, a badge, shirts |
| thr 0.5, vehicle 0.2, no crops | 37 | 6 | the same plus the laptop through the low vehicle threshold |
| thr 0.35, vehicle 0.2, no crops | 57 | 6 | more laptop, keyboard, notepad |
| thr 0.35, vehicle 0.2, **crops** | 130 | 62 | **the real plate and the dealer plate in every frame they are visible** (62 hits at 0.85-0.92); plus paper, licence, laptop, keyboard, door decal, QR code |
| thr 0.35, vehicle 0.35, crops | 102 | 62 | the same plates; the laptop gone again; 14 hits between 0.35 and 0.5 are keyboard / decal / notepad |
| **shipped: thr 0.4, vehicle 0.35, crops** | **96** | **62** | as the row above minus six of the weakest junk hits |

So the crop pass is the change that matters: the two plates in the clips went from 7 confident
hits to 62 at the same threshold, which is what makes a plate *track* (2+ hits) form and hold.
Precision is unchanged in kind: what remains is plate-shaped things on vehicles (a sheet of
paper on a hood, a driver's licence held beside a car, a door decal), deleted with one key.
Recall on real traffic stops is still Wyatt's test: the bench clips have two plates.

Enhancement (bench/results/ENHANCE.md) was also tried on the plate detector: boosting every
frame gained 65 plate hits and lost 66, and the gained ones were all the in-car tablet, the
laptop and a "SERVICE" sign; in the shipped `auto` mode (dark or flat frames only) it is a
no-op on these daylight clips.

## Decision of 2026-11-02: ship the open-image-models YOLOv9-t, gated

Wyatt accepted the author's MIT grant (the way OpenScrub and PrivacyCam ship the same file;
the provenance caveat above stays on record in docs/LICENSES.md). The file is in the model
registry as `plate_yolov9t` with its SHA-256 (`888397b9…`, identical upstream and in
PrivacyCam's bundle), bundled by the installer, 20 ms per 720p frame on the CPU.

Precision is the problem the first evaluation found, so two filters sit in front of it
(`vision/detectors/plates.py`):

* PrivacyCam's shape filters: aspect ratio 1.1-8, boxes under 6x3 input px or over 35 % of
  the input dropped, at most 20 per frame, 10 % extra height each side.
* **Vehicle gate**: a plate is kept only when at least 80 % of it lies inside a COCO vehicle
  box (car, truck, bus, motorcycle) from the YOLOX detector that runs in the same pass. A
  plate held in the hand, or on a vehicle YOLOX misses, needs a drawn box.

Measured on the 12 bench clips, every 15th frame (1 080 frames), `bench/eval_plates.py`:

| | detections | per frame | ms / frame | what the contact sheet shows |
|---|---|---|---|---|
| raw model, threshold 0.2 | 397 | 0.368 | 19.8 | in-car laptop screen and keyboard in hundreds of frames, 9/11 shirts, notepads, a driver's licence, the Axon timestamp overlay (0.55) |
| shipped pipeline (threshold 0.5 + shape filters + vehicle gate) | 22 | 0.020 | 37.6 (YOLOX-tiny vehicles + plates) | a dealer "USA" plate on a car (6 frames), one real plate (0.69), a driver's licence held beside a car (0.79), a white sheet of paper on a vehicle (8 frames), a "GEAR" badge, a QR code, two shirts |

The gate removes the laptop screen entirely (it is never inside a vehicle box when filmed from
the driver's seat). What remains is plate-shaped things on vehicles; the sheet of paper is the
kind of false hit the reviewer deletes with one key. Raising the threshold does not separate
them (the paper scores 0.70-0.84, the real plate 0.69), so the threshold stays at 0.5 and the
track-level rule (2+ detections) does the rest. The clips contain few plates, so recall is
still unmeasured: the gate is Wyatt's traffic-stop footage with visible plates, scored with
`bench/track_eval.py --classes license_plate` once reference projects exist.

## Decision of 2026-10-29 (superseded above)

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
