# Video redaction rework (0.1.15): high-confidence auto-detect, manual boxes that hold

Wyatt's brief (2026-10-28): auto-detect "goes overboard" and labels objects as faces or people;
only redact what we are confident about; a manually drawn box must cover the object before and
after the frame it was drawn on; hotkeys; audio untouched; the point is speed of review.

Everything below was measured on the 45-minute reference clip (`bench/reference`, Axon Body 2,
720p, night, CJIS: local only) using the tracks saved by `bench/run_bench.py` and contact sheets
of one crop per track (`bench/reference/track_bins*`, built by a scratch script).

## 1. What auto-detect was producing (recall profile, 0.1.14 and earlier)

2,313 tracks: face 1,651 · person 422 · phone 193 · document 24 · screen 23. Face tracks binned
by how they were confirmed, with the share of real faces judged from a 48-crop contact sheet per bin:

| Face bin | Tracks | Track-frames | Real faces (by eye) |
|---|---|---|---|
| head-fallback (head box derived from a person box) | 691 | 147,811 | ~1 in 3 |
| 1 detection | 508 | 45,594 | ~45 % |
| 2 detections | 125 | 13,972 | ~35 % |
| 3+ detections, avg conf < 0.6 | 60 | 12,576 | ~35 % (dark silhouettes, phones) |
| 3+ detections, avg conf 0.6-0.7 | 59 | 13,555 | ~40 % |
| 3+ detections, avg conf 0.7-0.8 | 30 | 6,893 | ~85 % |
| **3+ detections, avg conf >= 0.8** | **134** | **62,896** | **~100 %** |
| box taller than half the frame | 44 | 3,878 | ~5 % (walls, phones, people) |

The false "faces" were phones, lights, hands, car parts, the Axon overlay and dark blobs.
All 422 whole-body "person" tracks were unrequested (a side effect of the head fallback).
Phones were the opposite case: YOLOX scores real phones 0.35-0.6 and even its single detections
were ~80 % real, so objects need a looser rule than faces.

## 2. The precise profile (new default)

`AutoDetectOptions` defaults, `confirm_track()` in `vision/auto_detect.py`:

| Rule | Faces | Phones / screens / documents |
|---|---|---|
| Minimum detections per track | 3 | 2 |
| Minimum average detection confidence | 0.70 | 0.40 |
| Box taller than 60 % of the frame | rejected | kept |
| Head-fallback boxes | off (option) | - |
| Whole-body person boxes | only when "person" is ticked | - |

The old behaviour is `AutoDetectOptions.recall_profile()`: the Auto-detect dialog's
"Only keep confident detections" box unticked, or `run_bench.py --detect-opt min_hits=1 ...`.

### Result on the reference clip

| | Recall profile (before) | Precise profile (now) |
|---|---|---|
| Tracks | 2,313 | **362** (face 250 · phone 96 · document 9 · screen 7) |
| Track-frames | ~520,000 | **119,592** |
| Face tracks by bin | see above | 187 at conf >= 0.8 (all real on the sheet), 63 at 0.7-0.8 (~90 % real: residual hits are the Axon overlay logo and two over-exposed blobs) |
| Phone tracks | 193 (105 single-hit) | 96, all real on the 2-hit sheet |
| Auto-detect speed, single process, iGPU tier | 92.7 fps | **126.5 fps** (fewer boxes to carry) |
| Hand-redacted notepad covered | 20/21 boxes, only by accidental person boxes | 0/21 (no person boxes; documents still need the detector work in REFERENCE_TANYARD.md) |

What is given up, deliberately: faces seen fewer than three times (people crossing the frame in
under half a second at stride 5), faces the detector never scores above ~0.7 on average (very
dark or turned-away heads), and the accidental coverage of the notepad. Wyatt asked for this
trade; the reference's own author redacted none of those faces either.

## 3. Manual boxes: cover before and after

`bench/reference` notepad, hand-redacted 665-688 s. A box drawn at 677 s, tracked with the
single-object tracker (`vision/tracker.py`, ViTTrack, every 2nd frame):

| | Before | After |
|---|---|---|
| Spans | 17 fragments | **1** |
| Covered | 664.2-686.2 s, 13.0 s of the 22 s | **663.9-686.5 s, 22.7 s** (human: 23 s) |
| Time, stop 60 s after the box | 4.1 s | 3.2 s |
| Time, "keep searching to the end" | 85.2 s, plus a false re-acquisition at 908 s | 4.4 s, stops 8 s after the pad is gone |

Changes: the last box is held for 12 frames after the tracker loses the object, gaps of up to 2 s
between re-acquisitions of a nearby box are interpolated over (`bridge_spans`), and tracking in
either direction stops once the object has been out of view for `lost_limit_s` (8 s; the dialog
offers 30 s or "to the end"). A face box drawn at 2450 s followed the subject back to 1014 s and
forward to the end of the clip in both versions (43 spans now instead of 181); that run costs
about 5 minutes of compute per 25 minutes of video and is the reason the default stops early.

## 4. Review keys

N / Shift+N next/previous redaction · Delete (Ctrl+Z undo, no confirmation) · E enable/disable ·
X split at playhead · [ ] trim · = extend · T track selected box · F / Shift+F flagged tracks.
A plain drag draws and tracks a box at once (last label/shape/style); Shift+drag asks first.

## 5. Standard 12-clip benchmark

See `COMPARISON.md` (row `phase6_precise.json`).
