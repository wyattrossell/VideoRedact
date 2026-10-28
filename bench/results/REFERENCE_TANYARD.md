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

Three fine-tuning runs of YOLOX-tiny on the OWLv2 pseudo-labels (`train/train_loop.py`, dev
laptop RTX 3050 Ti 4 GB). Run 3 is the honest one: the reference clip was held out of training
(train 4,101 frames / 1,690 boxes from the other clips; held-out 400 frames / 348 boxes), 416 px,
60 epochs, 125 min. Numbers below are from `train/eval_ckpt.py` on its best checkpoint.

| | document AP50 | plate AP50 |
|---|---|---|
| Training frames (can it fit its own labels? 400-frame sample) | **0.56** | 0.31 |
| Held-out reference clip (OWLv2 labels: 14 document, 334 plate boxes) | **0.00** | 0.02 |
| Notepad frames 666 s - 687 s, best "document" score | 0.05, and that box is on the wall, not the pad | - |
| Highest "document" scores anywhere in the 45 min | 0.55 / 0.29 on car grilles at ~40 min | - |

So the student memorises the scenes it was trained on and does not transfer to a new one.
Two causes, both measured:

1. **The teacher is the ceiling.** OWLv2 scores the pad 0.13 - 0.34 ("a sheet of paper") on the
   notepad frames, but a framed photo on a counter at 900 s scores 0.22, so the 0.24 labelling
   threshold caught only part of the span - in fact **0 of the 14 held-out document labels lie
   inside 665 s - 688 s**. Lifting the dark frames first (CLAHE + gamma 0.6,
   `bench/eval_openvocab.py --enhance`) lowers the scores rather than raising them.
2. **Too few scenes.** 4,101 frames come from about ten clips; a 416 px tiny model fits them
   (0.56) and has nothing general to learn from. Runs 1 and 2 (640 px, 20 epochs; v1 labels)
   did no better: held-out AP50 0.006 - 0.017.

**Decision: not shipped.** No `yolox_tiny_doc.onnx` goes into the installer; the ensemble hook
(`doc_model` in the detector config) stays in the code and is inert without the file, so the
shipped detector is unchanged and the benchmark numbers in `COMPARISON.md` still apply.

What would change the outcome (needs footage we do not have on the dev laptop):

- a few hundred **hand-drawn** boxes on paper / notepads / forms across 20+ scenes and lighting
  conditions (the Label Studio workflow in `bench/README.md` §3-4 pre-annotates with OWLv2 at a
  low threshold so a reviewer only accepts/rejects), then the same `finetune_yolox.py` run;
- or treating the "officer's own notepad" case as a UI feature rather than a detector: one
  manual box plus the tracker already follows the pad for the 23 s (that is how the reference
  was probably made).

## Where VideoRedact stands against this reference

- **Leak rate:** 20 / 21 boxes covered, but by accident (person boxes). With documents-only
  policy and no person boxes it is 0 / 21. The human's output cannot be reproduced by
  auto-detect today; it can be reproduced in about a minute with one manual box + tracking.
- **Over-redaction:** 17,172 extra box-frames. Everything the human left visible (four faces for
  45 min, a phone, plates) we would blur by default. Per-class selection already exists; a
  "documents and plates only" preset is a one-line policy change (see decision for Wyatt below).
- **Review burden levers measured on the saved tracks** (`bench/tracks`, `run_bench.py --tracks-from`):

  | Setting | Tracks | Note |
  |---|---|---|
  | current defaults | 2,313 | |
  | `emit_person_boxes: false` | 1,891 | drops reference coverage 95 % -> 0 % (coverage was accidental) |
  | + drop single-detection tracks | 1,225 | |
  | + require 3 detections | 841 | not measured for leak yet |
  | + link fragments across 40 s gaps | 565 | **rejected**: face track-minutes 171 -> 397, interpolation paints faces where there are none |

  Long-gap linking needs identity, not position: see the identity grouping result below.

## Identity grouping prototype (`bench/face_identities.py`)

Can we tell the reviewer "these 1,651 face tracks are N people" so a policy like "everyone except
the officer" or "only the subject" is one click? SFace (OpenCV Zoo, Apache-2.0) embeddings of up
to three samples per face track, YuNet re-detection inside a padded crop, average-linkage
clustering on cosine distance. 7.3 min for the 45-min clip on the dev laptop (CPU), 0.2 s of
that is clustering.

| | |
|---|---|
| Face tracks | 1,651 |
| Tracks with at least one usable embedding | **966 (59 %)**; the other 685 are faces too small or too dark for YuNet to re-find in the crop |
| Clusters at the published SFace match point (distance 0.64) | ~230 "people" - far too fragmented |
| Clusters at distance 0.70 | 94 "people"; the three largest (249, 149, 85 tracks = 38.5, 22.9, 9.9 min on screen) are each **one real person** on their contact sheets, and together they hold half of the embedded tracks |
| Clusters at distance 0.40 | 632 "people", pure but useless (one person split into dozens) |

Read: grouping works for the people who matter (the ones on screen for minutes), and it would
let the reviewer keep or drop a person with one action instead of hundreds. It cannot group
the 41 % of tracks that are too small or dark to embed - exactly the ones that are also the
most numerous fragments - so it reduces burden for the big tracks, not the long tail. Not in
the product yet; it needs a UI (people panel with contact sheets, keep/redact per person) and
the officer should be identified by the user, never automatically.

Contact sheets and the per-threshold JSON are in `bench/reference/identities*` (local only).

## Decision needed from Wyatt

`emit_person_boxes` (auto-detect option, default `true`): turning it off removes 422 whole-body
tracks (18 % of all tracks, 183k track-frames) that nobody asked for, but on this reference it
is the only thing that covered the notepad. Recommendation: flip the default **together with** a
document detector that works, not before. Until then it is a per-run option.

## Notes on the method

- The diff must tolerate a global tone change inside the edited segment (histogram matched per
  frame), the one-frame timing mismatch between 29.97 and 30 fps (minimum over neighbouring
  originals) and blur boxes over dark, textureless areas (weak but area-consistent change plus
  texture loss, gated on the copy being flat). Thresholds were tuned on this pair only.
- Pair B ("Pioneer Trail") is incomplete: we have the redacted part 1 (91 min) and the original
  part 2 (4 min). Its redacted copy also leaves the officer visible. Needs the original part 1.
