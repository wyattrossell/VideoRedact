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

Precision rework vs Phase 4: auto-detect +47 % (fewer boxes to carry between detections), export
unchanged. Leak rate on labelled footage is still unmeasured; on the reference pair the only
hand-redacted object (a notepad) was covered before only by unrequested whole-body boxes, so
that accidental coverage is gone by design (Wyatt's 2026-10-28 brief: confident detections only).

Phase 4 vs baseline: auto-detect +40 %, export −2 % (the manifest and the second source hash add
about a second per clip; within run-to-run noise). Phase 4 vs Phase 3 iGPU: +33 % auto-detect from
the Phase 3 stream/adaptive-stride work landing after that measurement, on the larger clip set.

The first Phase 4 benchmark run crashed in the manifest writer on float32 box coordinates from
auto-detected tracks (fixed in 0.1.11, `tests/test_audit.py::test_manifest_accepts_numpy_boxes`).
That is exactly what the "rerun the benchmark after each phase" rule is for.

CI (`.github/workflows/ci.yml`) runs the same harness on the synthetic `tests/data/sample.mp4` on
every release and attaches `ci_<tag>.json` to the release; those numbers are for a GitHub CPU
runner and are not comparable with this table.
