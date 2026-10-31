# Phase 4: "track anything" segmentation on the CPU (2026-10-31)

Gate set on 2026-10-29: a promptable video-segmentation model is worth integrating only if it
propagates at 5+ fps at about 480p on the 8-core target, with no PyTorch in the installer.

## Survey (research agent, 2026-10-29, web only)

| Model | Licence | Full video pipeline in ONNX? | CPU evidence |
|---|---|---|---|
| **EdgeTAM** (Meta, 2025) | Apache-2.0 | Yes: `jax-image-tools/edgetam-video-onnx` (Apache-2.0, 4 graphs, memory bank assembled by the host, validated to IoU 1.000 against PyTorch) | one unofficial PyTorch number: 243 ms/frame at 512 px on 2 threads of an i7-12800HX |
| SAM 2.1 tiny / small | Apache-2.0 | community exports | ~0.5 fps at 1024 on a 6-thread laptop; memory attention over 28k tokens dominates |
| EfficientTAM | Apache-2.0 | no export exists | none |
| sam2-lite, TinySAM 2, SAM 3 | non-commercial / unpublished / custom licence | - | excluded |

EdgeTAM is the only candidate. Its Spatial Perceiver compresses each memory frame from 4096 to
512 tokens, which is what makes CPU propagation thinkable at all.

## Measured here (`bench/bench_edgetam.py`, ONNX Runtime CPU, i7-14700T, build running alongside)

The export is fixed at 1024 x 1024 input. Per-graph latency with a full memory bank (7 spatial
memories x 512 tokens + 16 object pointers):

| Graph | ms per run |
|---|---|
| vision encoder (1024 px) | 167 |
| mask decoder | 53 |
| memory attention (4096 current tokens vs 3 584 + 64 memory tokens) | 283 |
| memory encoder | 97 |
| **one propagated frame** | **600 ms, 1.7 fps** |

Two to three times short of the gate even before decode, mask-to-box conversion and the host loop.
A 512 px re-export would cut the encoder, decoder and memory encoder by about 4x and the
attention's current tokens to 1 024 (memory to 7 x 128), so roughly 120-180 ms per frame is
plausible (5-8 fps), but that needs a PyTorch + transformers re-export in the training venv, a
hand-written host loop for the memory bank, and the paper's own numbers say small, thin objects
lose the most accuracy at 512 (DAVIS J&F 86.4 -> 80.1). Distant licence plates at 1080p are exactly
that regime.

## Decision

Phase 4 does not pass its gate with the export that exists. Not integrated. Revisit only if
(a) a 512/768 export is produced and measured at 5+ fps here, and (b) it beats the anchored
tracker on Wyatt's hand-redacted ground truth (`bench/score_project.py --retrack`) on plates,
where the box tracker is weakest. Until then the product's answer for a hard object is the
anchored tracker plus the join key, which costs the reviewer one extra box, not a 10x slower
pass.
