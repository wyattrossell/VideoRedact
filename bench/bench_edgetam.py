"""EdgeTAM on the CPU: per-graph latency of the Apache-2.0 ONNX export (Phase 4 gate).

The export (Hugging Face jax-image-tools/edgetam-video-onnx) has four graphs; the memory bank
is assembled by the host. This script inspects each graph's inputs and outputs, feeds tensors
of the declared shapes (dynamic memory axes filled to a full bank: 7 spatial memories x 512
tokens + 16 object pointers x 4 tokens, from constants.json) and times them on ONNX Runtime
CPU. The sum is the floor for one propagated frame at the export's fixed 1024 px input.

Usage: python bench/bench_edgetam.py <dir with the four .onnx + constants.json> [--runs 10] [--threads 0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    import onnxruntime as ort
    consts = json.load(open(os.path.join(a.dir, "constants.json")))
    so = ort.SessionOptions()
    if a.threads:
        so.intra_op_num_threads = a.threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    total = 0.0
    outputs = {}
    for name in ("vision_encoder", "mask_decoder", "memory_attention", "memory_encoder"):
        path = os.path.join(a.dir, name + ".onnx")
        t0 = time.perf_counter()
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        load_ms = (time.perf_counter() - t0) * 1000
        print(f"\n== {name}  ({os.path.getsize(path) / 1e6:.1f} MB, loaded in {load_ms:.0f} ms)")
        feeds = {}
        for i in sess.get_inputs():
            shape = []
            for k, d in enumerate(i.shape):
                if isinstance(d, int):
                    shape.append(d)
                else:
                    ds = str(d)
                    # dynamic axes: fill to a full memory bank
                    if "spatial" in ds or "mem" in ds.lower() and "point" not in ds.lower():
                        shape.append(consts["num_maskmem"] * consts["spatial_tokens_per_memory_block"])
                    elif "point" in ds.lower() or "ptr" in ds.lower():
                        shape.append(consts["num_object_pointers_traced"] * consts["tokens_per_pointer"])
                    elif "batch" in ds.lower():
                        shape.append(1)
                    else:
                        shape.append(1 if k == 0 else 4)
            dtype = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(int64)": np.int64,
                     "tensor(int32)": np.int32, "tensor(bool)": np.bool_}.get(i.type, np.float32)
            arr = (np.random.rand(*shape).astype(dtype) if dtype in (np.float32, np.float16)
                   else np.zeros(shape, dtype))
            feeds[i.name] = arr
            print(f"   in  {i.name:32} {i.type:16} {i.shape} -> fed {list(arr.shape)}")
        for o in sess.get_outputs():
            print(f"   out {o.name:32} {o.type:16} {o.shape}")
        try:
            sess.run(None, feeds)                       # warm-up
            t0 = time.perf_counter()
            for _ in range(a.runs):
                outs = sess.run(None, feeds)
            ms = (time.perf_counter() - t0) * 1000 / a.runs
            outputs[name] = [(o.name, list(v.shape)) for o, v in zip(sess.get_outputs(), outs)]
            print(f"   {ms:.1f} ms per run (mean of {a.runs})")
            total += ms
        except Exception as e:  # noqa: BLE001
            print(f"   run failed with the guessed shapes: {str(e)[:300]}")
    print(f"\nsum of the four graphs at 1024 px, full memory bank: {total:.0f} ms per propagated frame "
          f"({1000 / total:.1f} fps) on {a.threads or os.cpu_count()} threads")
    print("gate (2026-10-29): 5 fps at ~480p; the export is fixed at 1024 px, a 512/768 re-export needs torch")


if __name__ == "__main__":
    main()
