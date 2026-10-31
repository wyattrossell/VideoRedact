# Re-identification attack on exported redactions (2026-10-31)

Tool: `bench/reid_attack.py` (protocol after OpenScrub's `benchmark/reid.py`, re-implemented on
our detector, identity model and track model). Question answered: with the original in hand,
can SFace still match the people in the *redacted* file to the people in the original?

Footage: one 45-second 720p30 Axon clip from `bench/clips` (crash scene, several people close
to and far from the camera; CJIS, local only). Redactions: the shipped auto-detect profile for
faces (precise: 3+ detections at 70 %+), 20 tracks, exported once per style. Numbers:
`reid_attack_2026-10-31.json`.

## Protocol in one paragraph

Every 15th frame, YuNet on the *source* at threshold 0.30 proposes candidates; only faces
confirmed at 0.60 and at least 32 px across are attacked (133 candidates below that are counted,
not graded). Confirmed faces are IoU-linked into tracks (45). Control: original-vs-original
similarity within a track, 0.733 over 43 pairs, against 0.226 between different people, so the
recogniser does identify these people when nothing is redacted and a low number after redaction
means something. For each face SFace is aligned on the *source* landmarks and run on the
*redacted* pixels; a face whose original embedding matches a flat patch of its own mean colour
at >= 0.5 is excluded as colour mass (1 face). Faces no enabled track covered are reported as
"left visible" (7 face-frames, 8 %: faces the precise profile did not confirm, i.e. misses of
auto-detect on this clip, not reviewer policy) and not attacked.

## Result

| Style | Face-frames attacked | Pixels unchanged (miss) | Re-identified at 0.363 (app threshold) | At 0.55 | Still detected as a face | Mean similarity after |
|---|---|---|---|---|---|---|
| Black (default) | 80 | 0 | **0** | 0 | 0 | -0.02 |
| Pixelate (16 px block floor, 8 blocks across) | 80 | 0 | 3 | 0 | 0 | 0.17 |
| Heavy blur (6 samples across) | 80 | 0 | 4 | 0 | 1 | 0.14 |

Mean similarity after redaction is below the different-person level (0.226) in all three
styles: on average a redacted face matches its own original no better than a stranger's face.

The residual matches at the loose threshold, and what they are:

| Style | Time | Face px | Similarity | Flat floor of the original | Reading |
|---|---|---|---|---|---|
| blur / pixelate | 40.5 s | 117 | 0.50 / 0.52 | 0.01 | **Genuine residual**: a large face keeps enough low-frequency identity through 6 samples / 7 blocks to score 0.5 (below the 0.55 strict point; black scores 0.28 on the same face) |
| blur / pixelate | 1.0 s, 1.5 s | 114, 119 | 0.39-0.50 | 0.37-0.40 | The original embedding of these frames already resembles a flat patch at 0.4 (dark, low-contrast face); black scores 0.35 and 0.25 on them, so most of the "match" is colour mass, not identity |
| blur / pixelate | 41.0 s | 83 | 0.38 / 0.36 | 0.38 | Same: floor 0.38 |
| pixelate | 38.0 s | 35 | 0.35 | 0.29 | Just under the threshold; small face |

So one face-frame of 80 (1.3 %) is a real weak-redaction residual for blur and pixelate, at a
similarity the app's own tracker would call "same person" (0.363) but OpenScrub's point
(0.55) would not. Black has none.

## What follows

* Black stays the recommended and default style; the tester guide and CJIS statement say so.
* Blur and pixelate are measurably weaker on faces above ~100 px. If Wyatt wants them
  stronger, the knobs are `PIXELATE_BLOCKS` (8) and `BLUR_DOWNSCALE` (6) in
  `core/video_redact.py`; re-run this attack after changing them (a few minutes) and update
  docs/CJIS_COMPLIANCE.md section 5. Not changed without his say-so: it alters the look of
  every blurred screen and document too.
* The 7 face-frames left visible are the precise profile's misses on this clip, consistent
  with the 18 % unsupported overrun figure in TRACKING_V2.md: the gate for auto-detect recall
  is still Wyatt's hand-redacted projects.
* Re-run on every clip he hand-redacts: `bench/reid_attack.py --source X.mp4 --redacted
  X_REDACTED.mp4 --project X.vrproj`. A clip whose control is below 0.5 is refused, by design.
