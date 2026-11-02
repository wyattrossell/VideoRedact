# Detector-side brightness / contrast boost (2026-11-02)

Wyatt's request: "the auto redaction struggles due to lack of contrast ... temporarily edit the
video to provide the best possible brightness and contrast to make it easier for the software
to detect items. This should not be shown to the user but only to the program."

Implementation (`vision/enhance.py`): the luminance (L of Lab) is stretched so the darkest and
brightest half-percent of pixels sit at 0 and 255, then locally equalised (CLAHE, clip 2.0,
8x8 tiles); colour untouched. 7-11 ms per 720p frame. Only detectors see the copy; the
preview, the reviewer's frame and the export are the original pixels (two exports with the
setting on and off are byte-identical, `tests/test_enhance.py`).

Measured with `bench/eval_enhance.py` on the 12 bench clips, every 15th frame (1 080 frames):
the shipped pipeline is run on each frame twice, plain and boosted, and the detections are
matched per label (IoU >= 0.4). "gained" = found only with the boost, "lost" = only without.
The clips are daylight footage: mean grey 71-109, contrast (grey standard deviation) 60-85;
the darkest sampled frame has mean grey 31.

## Experiment 1: every frame boosted, for every detector

| label | both | gained | lost | score change on matched |
|---|---|---|---|---|
| face | 617 | 132 | 19 | +0.006 |
| license_plate | 16 | 19 | 6 | -0.012 |
| screen | 57 | 27 | **56** | -0.008 |
| phone | 8 | 5 | 8 | -0.009 |
| document | 3 | 22 | 0 | +0.056 |

Contact sheets (scratch only, CJIS): of the 132 gained "faces" roughly 90 are the Chevrolet
emblem on the steering wheel, the camera's own timestamp overlay, a wheel, hands and blobs, and
roughly 40 are real faces the plain pass missed (a bald man looking down, a man in sunglasses,
a bearded man in shadow, profiles). The 19 lost faces include real ones (a woman with glasses
at 0.72, a bearded man in profile). Screens: the boost flattens the bright in-car laptop screen
and YOLOX stops seeing it in half the frames it had (56 lost); the 27 gained screens are the
same laptop in other frames. The 22 gained "documents" are the laptop keyboard and the
notepad. Plates: the 19 gained are the in-car tablet's map screen and the Getac logo.

Verdict: a wash for real faces, a loss for screens, junk added everywhere. Not a default.

## Experiment 2: faces and plates only (YOLOX keeps the original), every frame

| label | both | gained | lost |
|---|---|---|---|
| face | 621 | 98 | 28 |
| license_plate | 127 | 65 | 66 |
| screen / phone / document | unchanged by construction | 0 | 0 |

Gained faces again dominated by the emblem (~25) and the timestamp overlay (~15); real gains
and real losses roughly balance (about 20 each). The gained plates are the tablet, the laptop,
a "SERVICE" sign, a police vest; the lost plates are of the same kind. The face track rule
(3+ hits at a 0.7 average) would reject most of the emblem / overlay hits, which score
0.50-0.64, but not all of the overlay's (up to 0.76).

By frame brightness, experiments 1 and 2 together: gained / lost on frames with mean grey
below 70: 28 / 15 (and 28 / 3 in experiment 1); between 70 and 130: 132 / 77; above 130:
3 / 2. The only bucket where the boost gains clearly is the dark one, and the clips have few
dark frames.

## What ships: a mode, default "dark or washed-out frames only"

`detect_enhance` = `off` | `auto` | `always` (Auto-detect dialog; also used by the detector
inside a drawn box's tracking). `auto` boosts a frame (or, in the tracker, the region around
the box) only when its mean grey level is below 60 or its grey spread below 30, measured on a
subsampled thumbnail. On daylight footage that is a no-op, which is the point: the measured
false hits came from boosting frames that did not need it. The face and plate detectors are
the only ones handed the copy; YOLOX (screens, phones, documents, people) and the export
verification keep the original (a boosted verification would list the emblem as an uncovered
face). `always` stays available for a clip that is dark throughout, which the bench clips are
not; that case is Wyatt's test.

## Experiment 3: the shipped `auto` mode on the same frames

`bench/eval_enhance.py --mode auto`: 32 of the 1 080 sampled frames (3 %) qualify as dark or
flat and get the boost; the rest are untouched.

| label | both | gained | lost |
|---|---|---|---|
| face | 649 | 8 | 0 |
| license_plate | 167 | 2 | 3 |
| screen / phone / document | 132 | 0 | 0 |

All 8 gained faces are on frames with mean grey below 70. No face is lost, and the cost on
daylight footage is the thumbnail statistic (under a millisecond per frame). That is the
default. How much `auto` or `always` helps on footage that is dark throughout (night stops,
indoor scenes) cannot be measured on these clips; it is the question for Wyatt's test.

