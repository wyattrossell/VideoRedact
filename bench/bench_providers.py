"""ONNX Runtime provider benchmark for our detectors (run in a venv that has the
provider installed: onnxruntime (CPU), onnxruntime-openvino, onnxruntime-gpu).

    python bench/bench_providers.py [--providers CPU,OpenVINO:CPU,OpenVINO:GPU,CUDA,TensorRT] [--batch 1,4,8]

Reports ms per frame for yolox_tiny (416), yolox_s (640) and yunet (1280x720 input)
with warm-up, and whether the model accepts a batch > 1 (dynamic batch dim).
Engine caches (TensorRT / OpenVINO) are written under <models>/cache so the second
run shows the cached start-up time.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

MODELS = Path(os.environ.get("LOCALAPPDATA", "")) / "VideoRedact" / "models"
FILES = {"yolox_tiny": ("yolox_tiny.onnx", (1, 3, 416, 416)), "yolox_s": ("yolox_s.onnx", (1, 3, 640, 640)),
         "yunet": ("face_detection_yunet_2023mar.onnx", (1, 3, 736, 1280))}


def providers_for(name: str, cache: Path):
    if name == "CPU":
        return ["CPUExecutionProvider"]
    if name.startswith("OpenVINO"):
        dev = name.split(":")[1] if ":" in name else "CPU"
        return [("OpenVINOExecutionProvider", {"device_type": dev, "precision": "FP16" if dev == "GPU" else "FP32",
                                              "cache_dir": str(cache / "openvino")}), "CPUExecutionProvider"]
    if name == "CUDA":
        return [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    if name == "TensorRT":
        return [("TensorrtExecutionProvider", {"trt_fp16_enable": True, "trt_engine_cache_enable": True,
                                             "trt_engine_cache_path": str(cache / "tensorrt")}),
                ("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    raise ValueError(name)


def main():
    import onnxruntime as ort
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", default="CPU")
    ap.add_argument("--batch", default="1,4")
    ap.add_argument("--iters", type=int, default=30)
    a = ap.parse_args()
    cache = MODELS / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    print("onnxruntime", ort.__version__, "available:", ort.get_available_providers())
    for pname in a.providers.split(","):
        try:
            provs = providers_for(pname, cache)
        except ValueError:
            print(f"{pname}: unknown")
            continue
        for mname, (fname, shape) in FILES.items():
            path = MODELS / fname
            if not path.exists():
                print(f"  {mname}: model missing at {path}")
                continue
            so = ort.SessionOptions()
            so.log_severity_level = 3
            so.intra_op_num_threads = max(1, min(4, (os.cpu_count() or 4) // 2))
            so.add_session_config_entry("session.intra_op.allow_spinning", "0")
            t0 = time.perf_counter()
            try:
                sess = ort.InferenceSession(str(path), so, providers=provs)
            except Exception as e:  # noqa: BLE001
                print(f"{pname:14s} {mname:10s}: session failed: {str(e)[:120]}")
                continue
            load = time.perf_counter() - t0
            used = sess.get_providers()[0]
            inp = sess.get_inputs()[0]
            dyn_batch = not isinstance(inp.shape[0], int)
            for b in (int(x) for x in a.batch.split(",")):
                if b > 1 and not dyn_batch:
                    print(f"{pname:14s} {mname:10s} batch {b}: static batch dim ({inp.shape}) - skipped")
                    continue
                x = np.random.randint(0, 255, (b,) + tuple(shape[1:]), dtype=np.uint8).astype(np.float32)
                try:
                    for _ in range(3):
                        sess.run(None, {inp.name: x})
                    t0 = time.perf_counter()
                    for _ in range(a.iters):
                        sess.run(None, {inp.name: x})
                    ms = (time.perf_counter() - t0) / a.iters / b * 1000
                    print(f"{pname:14s} {mname:10s} batch {b}: {ms:7.1f} ms/frame  (provider {used}, load {load:.1f}s)")
                except Exception as e:  # noqa: BLE001
                    print(f"{pname:14s} {mname:10s} batch {b}: run failed: {str(e)[:100]}")


if __name__ == "__main__":
    main()
