# Tracking rebuild, Phase 1 (2026-10-29): the anchored single-object tracker

Wyatt's brief (2026-10-29): a face or plate he boxes by hand tracks for a few frames and stops,
so he has to box the same object several times in one clip. Rebuild tracking so one box covers
the object's whole visible life around the drawn frame, before and after.

## What was wrong (measured before any change)

`bench/track_eval.py` draws a box in the middle of a confident auto-detected face track (precise
profile, so the track is real) and runs the single-object tracker exactly as the app does: both
directions, every 2nd frame, 8 s lost limit. Coverage = share of the auto track's frames the
manual track covers. 24 face cases from the 720p/1080p clips in `bench/clips`, plus the 4 phone
tracks long enough to use as a rigid-object proxy.

| Engine 0.1.15 ("classic", `vision/tracker.py`) | Coverage | Cases >= 90 % |
|---|---|---|
| face, with the face detector | 86.1 % | 18/24 |
| face, no detector (how a plate or tattoo is tracked today) | 76.2 % | 14/24 |

Why (from per-frame traces): the tracker worked on a 512 px copy of the frame (a 72 px face is 29 px
to it and to the detector it snaps to), compared every frame with the original patch and nothing
else, gave up when ViTTrack's own score dipped below 0.30, and re-acquired only through a
template match against that original patch. Variants tried on the old engine: drift check off
90 %, score cutoff 0.15 91 %, 1024 px work frame 89 % (2.5x slower). No knob fixed it.

## The anchored engine (`vision/anchored.py`, default for detector classes)

* Full resolution. ViTTrack crops its own search window, so its cost does not depend on frame size.
* The class detector runs on a region around the predicted box (YuNet on the CPU ONNX runtime:
  1-2 ms per crop, 11 ms per full frame; the Intel GPU path recompiles per crop size, 450 ms, so
  the tracker's detector stays on the CPU), on a schedule and immediately whenever the tracker is
  unsure, failed, or produced an implausible box. Small objects are upscaled before detection.
  Detector threshold 0.30 while tracking: the region, size and identity gates do the filtering.
* Appearance bank of up to 8 confident views (inner 70 % grey template + colour histogram), plus an
  SFace identity embedding for faces (OpenCV Zoo, Apache-2.0, 20 ms per embedding, same person
  0.52-0.62 vs other people 0.2-0.28 on our frames; threshold 0.363, 0.25 close to the loss point).
* Loss with hysteresis: detector missing 3 checks while the tracker is unsure; faces also after
  6 checks or 3 s unconfirmed; the box leaving the frame; a box ballooning past 3x the last confirmed
  area. The unconfirmed tail before a loss is dropped, the last confirmed box held 12 frames.
* Re-acquisition while lost: detector candidates gated by plausible size and position for the time
  lost (frame-wide after 1 s), then identity (faces) or appearance; other classes may also
  re-acquire by template / colour search. Gaps under 1 s are bridged with a motion-aware distance.
* ViTTrack's internal failure cut lowered from 0.20 to 0.05 so the engines see every box and
  apply their own hysteresis.

### Results (same 24 face cases and 4 phone cases, `bench/results/track_eval_*.json`)

| | classic | anchored | 
|---|---|---|
| face with detector: coverage | 86.1 % | **97.1 %** |
| face with detector: cases >= 90 % | 18/24 | **21/24** |
| face with detector: overrun frames beyond the auto track, and the share of sampled overrun frames where neither an auto track nor a 0.30 probe detector finds a face | 7 759 (11 %) | 11 722 (18 %) |
| phone with detector: coverage | 86.9 % | **87.4 %** (overrun 1 524 -> 522 frames) |
| face / phone without a detector | 76.2 % / 86.9 % | routed to the classic loop (same numbers) |

The overrun is mostly real coverage: the precise auto profile ends a track when the detector stops
seeing a face, and the manual tracker keeps covering it. The contact sheet of "unsupported"
overrun frames (scratch, not committed) is dominated by small, blurred or half-visible faces the
0.30 probe cannot see; the real drift cases are the Axon overlay (a face passing under it), a hand
holding a notepad, and a "POLICE" vest, each for a few seconds before the loss rules fire.

### What did not work: the anchored engine without a detector

Faces tracked as "other" (no detector) are the proxy for plates and tattoos. Seven variants of the
no-detector rules measured 65-77 % coverage, never better than the classic loop's 76.2 %, and the
ones that matched it did so with 4x its drift (53 % vs 12.5 % unsupported overrun samples). The
per-step box sanity check and the appearance checks that stop drift also stop legitimate
appearance change; a detector is what makes the anchored approach work. Decision: objects without
a class detector keep the classic loop (`track_engine: anchored`); `anchored-all` forces the new
engine for them (it can re-acquire a flat object that left the frame, which the classic loop
cannot: `tests/test_anchored.py`). Plates get their detector in Phase 2.

Gate set on 2026-10-29: >= 95 % of confident faces with the detector (met: 97.1 %), >= 85 % without
(not met by either engine on faces-as-other: 76.2 %; phones-as-other 86.9 %).

### Speed

Anchored, faces with detector, 24 cases both directions: 126 s for 24 cases on an idle machine
(classic 108 s). Per case the heaviest 1080p run covered 1 348 frames in 18.7 s (72 fps, 2.4x
real time) while three runs shared the CPU; typical 720p cases run 3-6 s for 300-900 frames.

## Also fixed on the way

* Auto-detect on 1080p sources ran on a frozen picture on this laptop (QuickSync); see
  `CHANGELOG.md` 0.1.16 and `COMPARISON.md`.
* The tracker's YuNet now comes with landmarks (ONNX adapter decodes `kps_*`), which SFace needs.

## Next

* Wyatt's hand-redacted projects as ground truth (`bench/score_project.py` to come) replace the
  auto-track proxy, especially for plates and notepads.
* Phase 2: licence-plate detector (`bench/results/PLATES.md`).
* Phase 3: joining a drawn box to the selected track (shipped with this phase: J, or draw in a gap).
