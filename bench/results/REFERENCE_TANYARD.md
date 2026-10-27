# Reference pair A ("Tanyard", 45.6 min welfare check, Axon Body 2, 720p) vs VideoRedact

Hand-redacted copy supplied 2026-10-26 (another agency's tool, re-encoded 29.97 -> 30.00 fps,
same duration). Ground truth derived with `bench/labels_from_reference.py` (one sample per
second, 2,736 samples). Footage and labels stay on the dev laptop (CJIS).

## What the human redacted

| | |
|---|---|
| Redacted spans | **one**: 665 s - 688 s (23 s), a notepad / form on the table that an officer is writing on |
| Style | blur box (flat grey), drawn generously around the pad (median box 225 px tall) |
| Faces | **none** redacted: officer, subject and two bystanders stay visible for the full 45 min |
| Screens / phones / plates | none redacted (a phone appears at ~25 min, parked cars with plates at ~37 min) |
| Tone | the re-rendered 23 s segment has a global brightness shift (~13 levels); the rest is a bit-faithful re-encode |

So this reference encodes a **policy**: redact the written personal information, nothing else.
That is a very different target from "every face" and it exposes a real gap in our detector.

## VideoRedact 0.1.12 on the original (defaults: face, screen, document, phone; stride 5)

| Metric | Value |
|---|---|
| Auto-detect speed | 92.7 fps single process (14.7 min for 45.6 min of video), Intel iGPU tier |
| Tracks produced | **2,313** (face 1,651 · person 422 · phone 193 · document 24 · screen 23) |
| Reference boxes covered >= 90 % | 20 / 21 (95 %), **but only by whole-body "person" boxes** that happen to overlap the pad |
| Reference boxes covered by a *document* detection | **0 / 21** |
| Same detections without the person boxes | 0 / 21 covered |
| Extra boxes on labelled frames (what a reviewer following this policy must delete) | 17,172 box-frames on 2,444 of 2,736 sampled seconds |

Findings:

1. **The COCO "book" class does not see notepads.** YOLOX-tiny/-s give "document" at most 0.04
   confidence on the pad frames at any threshold. Every tool built on COCO classes has this gap.
2. **Whole-body person boxes were leaking into the output** even though the user only asked for
   faces: the head-fallback adds the person class to find heads and passed the person boxes
   through. 422 tracks / 183k track-frames of pure review burden. Now an option
   (`emit_person_boxes`); the default is unchanged until Wyatt approves, because on this
   reference the accidental person coverage is what kept recall at 95 %.
3. **Review burden is the real difference**, not speed: 2,313 tracks for one 45-minute file,
   a third of them single-detection fragments, when the human wanted exactly one box. Policy
   tools ("documents only", "everyone except the officer", per-person grouping) matter more
   than another percent of face recall for this workflow.

## What is being done about it

- `train/pseudolabel_openvocab.py`: OWLv2 (Apache-2.0) labelled 4,501 frames from all local
  footage for `document` and `plate` in 27 min on the dev laptop's GPU (1,637 + 2,323 boxes;
  noisy, reviewable in `train/pseudo/review`). OWLv2 itself is 7.4 s/frame on CPU, so it is the
  teacher, not the product.
- `train/finetune_yolox.py`: YOLOX-tiny fine-tuned on those labels -> `yolox_tiny_doc.onnx`,
  joins the detector ensemble automatically when present (`vision/detectors/__init__.py`), at
  COCO-detector cost (~10 ms per call on the iGPU).
- Scoring of the fine-tuned model against this reference: see the table below (filled in after
  training).

## Fine-tuned document detector vs reference

_(pending)_

## Notes on the method

- The diff must tolerate a global tone change inside the edited segment (histogram matched per
  frame), the one-frame timing mismatch between 29.97 and 30 fps (minimum over neighbouring
  originals) and blur boxes over dark, textureless areas (weak but area-consistent change plus
  texture loss, gated on the copy being flat). Thresholds were tuned on this pair only.
- Pair B ("Pioneer Trail") is incomplete: we have the redacted part 1 (91 min) and the original
  part 2 (4 min). Its redacted copy also leaves the officer visible. Needs the original part 1.
