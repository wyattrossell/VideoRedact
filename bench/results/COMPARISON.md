# Benchmark history (dev laptop: i7-14700T, UHD 770 iGPU, 64 GB; department clips, local only)

Same 11-12 × 45 s body-cam clips (`bench/clips`, git-ignored), stride 5, labels
face/screen/document/phone. Accuracy columns stay empty until ground-truth labels exist
(`bench/README.md` labelling workflow); **leak rate therefore cannot be compared yet**, so the
working rule "revert anything that raises leak rate" is checked with the reference-coverage
measure in `bench/eval_autodetect.py` and the post-export verification pass instead.

| Run | Version | Tier | Clips / frames | Auto-detect fps | Parallel fps | Export fps | Notes |
|---|---|---|---|---|---|---|---|
| baseline.json | 0.1.7 | CPU | 11 / 14 828 | 59.8 | 60.5 | 202.4 | before Phase 2 |
| phase3_cpu.json | 0.1.8 | CPU | 3 / 4 044 | 42.5 | – | 182.3 | Phase 2 accuracy pipeline (multiscale, head fallback, BoT-SORT) costs detection time |
| phase3_openvino.json | 0.1.8 | Intel iGPU | 3 / 4 044 | 63.0 | – | 180.2 | OpenVINO runner |
| **phase4.json** | **0.1.10** | Intel iGPU | 12 / 16 176 | **83.8** | 84.8 | 198.3 | Phase 3 streams + adaptive stride, Phase 4 audit (source re-hash + manifest per export) |
| **phase6_precise.json** | **0.1.14+** | Intel iGPU | 12 / 16 176 | **123.4** | 123.4 | 198.2 | Precision rework: confident-only tracks (3+ hits at 0.7+ for faces), no person/head-fallback boxes; 99 tracks / 18 939 track-frames vs 829 / 127 744 on the same clips (`PRECISION_PROFILE.md`) |
| release_0_1_19.json | 0.1.19 (pre-release) | Intel iGPU | 12 / 16 176 | 103.3 | 104.4 | 164.3 | Same labels as phase6 (face/screen/document/phone); 101 tracks / 18 013 track-frames. The lower numbers are the day, not the code: see the A/B note below |
| release_0_1_19_plates.json | 0.1.19 (pre-release) | Intel iGPU | 12 / 16 176 | 86.6 | 85.7 | 185.6 | Plus `license_plate` (YOLOv9-t behind the vehicle gate): detection time +42 % (152.9 s vs 107.6 s over 3 235 detector calls, ~14 ms per call), 9 plate tracks on the 12 clips (110 tracks / 18 652 track-frames) |
| release_0_1_20.json | 0.1.20 (pre-release) | Intel iGPU | 12 / 16 176 | 120.3 | 118.9 | 175.8 | Same labels; contrast boost in its default `auto` mode (boosts 3 % of the bench frames), zone gone, blur back: 101 tracks / 18 013 track-frames, identical to the 0.1.19 row. Faster than the morning's run (detection 96.9 s vs 107.6 s): day-to-day noise, nothing in the code got cheaper |
| release_0_1_20_plates.json | 0.1.20 (pre-release) | Intel iGPU | 12 / 16 176 | 63.9 | 62.6 | 175.1 | Plus `license_plate` with the round-two pipeline (plate model on the whole frame and on a crop of every detected vehicle): detection 210.7 s vs 152.9 s (+38 %, the crop passes), 15 plate tracks / 2088 plate track-frames against 9 in 0.1.19 (116 tracks / 20 101 track-frames). Precision and recall of those tracks: bench/results/PLATES.md "Round two" |
| release_0_1_20_margin40.json | 0.1.20 | Intel iGPU | 12 / 16 176 | 98.6 | 96.6 | 191.0 | Box margin 10 % -> 40 % of the box on each side (applied when a frame is rendered, nothing in detection): 101 tracks / 18 013 track-frames, identical to the two rows above. Auto-detect reads 18 % slower than yesterday, but the previous commit's code measured 98.7 fps (detect 115.1 s) in the same hour on this laptop, against 98.6 / 114.8-115.6 s for this code: the machine is slower today, not the code. Export 191.0 fps with the bigger boxes (redact stage 6.7 s vs 6.8 s): the extra pixels cost nothing measurable with the black-box style |

**0.1.20 (2026-11-02 pm):** the plate crop pass is the one measurable cost of this round:
auto-detect with plates drops from 86.6 to 63.9 fps on clips that have vehicles in nearly every
frame (each detected vehicle, up to four, is a second 384 px plate inference). Without plates
nothing changed in cost: the contrast boost in `auto` mode is a thumbnail statistic per
detection frame and boosted 32 of 1 080 sampled frames. The 15 plate tracks cover
2088 track-frames against 9 tracks before, on the same two plates (a real one and a dealer
plate on the stopped car; bench/results/PLATES.md). Leak rate still unmeasured (no labels).

**0.1.20 box margin (2026-11-03):** every new box renders half again bigger (margin 40 % of the
box on each side instead of 10 %; 1.8x the box's width and height instead of 1.2x) so a quick
pan of the body camera does not uncover a face. The margin is applied when a frame is rendered,
so detection output cannot change, and the run shows it: the same 101 tracks and 18 013
track-frames. The lower auto-detect fps in that row is the laptop, not the code: a detect-only
run of the previous commit in the same hour gave 98.7 fps (scratch run, not kept). Export got
no slower with the bigger boxes (191.0 fps, redact stage 6.7 s); the bench exports with the
default black-box style, a blur over 2.25x the area would cost more and was not measured.

**0.1.19 (2026-11-02) versus phase6, A/B on the first 3 clips (4 044 frames), the pre-session
commit (10f84eb) and HEAD run alternately in both orders:** auto-detect 101.4 / 109.3 fps before
versus 107.6 / 109.1 after (no change); export 210.9 / 199.9 before versus 161.7 / 159.1 after.
The export gap was traced to the FFmpeg binary, not the code: the old-commit worktree had no
`bin/` folder and launched the imageio-ffmpeg binary from local disk, while the repo launches
`binfmpeg.exe` from the network share, which costs about 0.3 s per FFmpeg process (an export
starts four: probe, audio decode, encode, two metadata reads). With `VIDEOREDACT_FFMPEG` pointing
HEAD at the local binary, a bare one-clip export took 3.9 / 5.7 s against 4.1 / 6.6 s before.
The installed app ships its FFmpeg locally, so agencies never pay this; the bench on a
redirected-desktop checkout does. Consecutive identical runs of the encoder on this laptop also
vary by up to 40 % (3.9 s and 5.7 s for the same clip), so export fps rows across days are
noise unless measured back to back. Leak rate is still unmeasured (no labels).

Precision rework vs Phase 4: auto-detect +47 % (fewer boxes to carry between detections), export
unchanged. Leak rate on labelled footage is still unmeasured; on the reference pair the only
hand-redacted object (a notepad) was covered before only by unrequested whole-body boxes, so
that accidental coverage is gone by design (Wyatt's 2026-10-28 brief: confident detections only).

**Invalid rows for `_1080p_test.mp4` in phase4.json and phase6_precise.json** (found 2026-10-29):
on this laptop the detection proxy pipe decoded that clip through QuickSync, which returned the
first frame for the whole clip, so its auto-detect numbers (179 fps, 9 static tracks) measure
nothing. Fixed in 0.1.16 (`core/streams.py` GUARD_FRAMES, d3d11va first); the 720p clips never
used the proxy path and their rows stand. A full re-run of the 12-clip set is part of Phase 1.

Phase 4 vs baseline: auto-detect +40 %, export −2 % (the manifest and the second source hash add
about a second per clip; within run-to-run noise). Phase 4 vs Phase 3 iGPU: +33 % auto-detect from
the Phase 3 stream/adaptive-stride work landing after that measurement, on the larger clip set.

The first Phase 4 benchmark run crashed in the manifest writer on float32 box coordinates from
auto-detected tracks (fixed in 0.1.11, `tests/test_audit.py::test_manifest_accepts_numpy_boxes`).
That is exactly what the "rerun the benchmark after each phase" rule is for.

CI (`.github/workflows/ci.yml`) runs the same harness on the synthetic `tests/data/sample.mp4` on
every release and attaches `ci_<tag>.json` to the release; those numbers are for a GitHub CPU
runner and are not comparable with this table.
