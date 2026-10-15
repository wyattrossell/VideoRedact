"""Inference runner abstraction: one call `run(blob) -> list[np.ndarray]` backed by
the best engine available on this machine.

Tiers (see vision/accel.py):
  * OpenVINO on an Intel GPU/NPU via the `openvino` Python API (Apache-2.0).
    Measured on an i7-14700T / UHD 770: YOLOX-tiny 11 ms (CPU 17), YOLOX-s 32 ms
    (CPU 83), YuNet@1280 8 ms, with the CPU left free for decode/tracking.
    Compiled models are cached on disk (CACHE_DIR) so start-up is fast after the
    first run. The ONNX Runtime OpenVINO *execution provider* is NOT used: its
    wheel is ABI-coupled to one OpenVINO release and failed to load against
    every published runtime (2026-10-15).
  * ONNX Runtime with the provider chain from accel.py (CUDA/TensorRT when the
    GPU pack is installed, else CPU).

`make_runner(model_path, dynamic_shape=True)` picks the engine; `runner.run(blob)`
returns the model outputs in the ONNX output order. Engines that fail to
compile fall back to ONNX Runtime CPU, so detection always works.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

from videoredact.paths import models_dir

log = logging.getLogger(__name__)
_OV_CORE = None
_OV_DEVICE: Optional[str] = None


def openvino_device() -> Optional[str]:
    """Intel GPU/NPU device name usable by OpenVINO, or None (cached)."""
    global _OV_CORE, _OV_DEVICE
    if _OV_DEVICE is not None:
        return _OV_DEVICE or None
    _OV_DEVICE = ""
    if os.environ.get("VIDEOREDACT_NO_OPENVINO"):
        return None
    try:
        import openvino as ov
        core = ov.Core()
        best = None
        for d in core.available_devices:
            try:
                name = str(core.get_property(d, "FULL_DEVICE_NAME"))
            except Exception:
                name = ""
            if d.startswith("NPU"):
                best = d
                break
            if d.startswith("GPU") and "intel" in name.lower():
                best = best or d
        if best:
            core.set_property({"CACHE_DIR": str(models_dir() / "cache" / "openvino")})
            (models_dir() / "cache" / "openvino").mkdir(parents=True, exist_ok=True)
            _OV_CORE, _OV_DEVICE = core, best
            log.info("OpenVINO device: %s", best)
    except Exception as e:  # noqa: BLE001
        log.info("OpenVINO unavailable: %s", e)
    return _OV_DEVICE or None


class ORTRunner:
    def __init__(self, path: str, threads: int = 0, outputs: Optional[list[str]] = None):
        from .detector import _ort_session
        self.sess = _ort_session(path, threads)
        self.input_name = self.sess.get_inputs()[0].name
        self.input_shape = self.sess.get_inputs()[0].shape
        self.output_names = outputs or [o.name for o in self.sess.get_outputs()]
        self.engine = f"onnxruntime/{self.sess.get_providers()[0].replace('ExecutionProvider', '')}"

    def run(self, blob: np.ndarray) -> list[np.ndarray]:
        return self.sess.run(self.output_names, {self.input_name: blob})


class OpenVINORunner:
    """Compiles once per input shape (dynamic models are reshaped to the first blob
    seen; later shape changes recompile, cached on disk). With nhwc_u8=True the
    model takes the raw BGR uint8 frame (HxWx3) and converts/transposes on the
    device, which removes a 10 ms CPU blob conversion per 1280x736 frame."""

    def __init__(self, path: str, device: str, outputs: Optional[list[str]] = None, precision: str = "f16",
                 nhwc_u8: bool = False):
        import openvino as ov
        self.ov = ov
        self.core = _OV_CORE
        self.device = device
        self.nhwc_u8 = nhwc_u8
        self.model = self.core.read_model(path)
        if nhwc_u8:
            from openvino.preprocess import PrePostProcessor
            ppp = PrePostProcessor(self.model)
            inp = ppp.input(0)
            inp.tensor().set_element_type(ov.Type.u8).set_layout(ov.Layout("NHWC"))
            inp.model().set_layout(ov.Layout("NCHW"))
            self.model = ppp.build()
        self.input_name = self.model.inputs[0].get_any_name()
        self.input_shape = [d.get_length() if d.is_static else -1 for d in self.model.inputs[0].get_partial_shape()]
        self.output_names = outputs or [o.get_any_name() for o in self.model.outputs]
        self.precision = precision
        self._shape = None
        self._req = None
        self._out_idx = None
        self.engine = f"openvino/{device}"

    def _compile(self, shape):
        model = self.model
        if any(d == -1 for d in self.input_shape):
            model = model.clone()          # keep the dynamic original for later shapes
            # after PrePostProcessor.build() the model input IS the NHWC u8 tensor, so the
            # tensor shape (n, h, w, c) is what gets reshaped
            model.reshape({self.input_name: self.ov.PartialShape(list(shape))})
        cfg = {"PERFORMANCE_HINT": "LATENCY"}
        if not self.device.startswith("CPU"):
            cfg["INFERENCE_PRECISION_HINT"] = self.precision
        comp = self.core.compile_model(model, self.device, cfg)
        self._req = comp.create_infer_request()
        names = [o.get_any_name() for o in comp.outputs]
        self._out_idx = [names.index(n) for n in self.output_names]
        self._shape = tuple(shape)

    def run(self, blob: np.ndarray) -> list[np.ndarray]:
        if self._shape != tuple(blob.shape):
            self._compile(blob.shape)
        res = self._req.infer({0: np.ascontiguousarray(blob)})
        outs = [res[k] for k in res]
        return [outs[i] for i in self._out_idx]


class FallbackRunner:
    """Wraps an accelerated runner; on any inference error switches to ONNX Runtime CPU for good."""

    def __init__(self, primary, path: str, threads: int, outputs):
        self.primary, self.path, self.threads, self.outputs = primary, path, threads, outputs
        self.engine = primary.engine
        self.input_shape = primary.input_shape
        self._fallback = None

    def run(self, blob: np.ndarray) -> list[np.ndarray]:
        if self._fallback is None:
            try:
                return self.primary.run(blob)
            except Exception as e:  # noqa: BLE001
                log.warning("%s failed (%s); switching to ONNX Runtime CPU", self.engine, e)
                self._fallback = ORTRunner(self.path, self.threads, self.outputs)
                self.engine = self._fallback.engine
                self.takes_nhwc_u8 = False
        if blob.ndim == 4 and blob.dtype == np.uint8:   # NHWC u8 arrived but the fallback wants NCHW f32
            blob = np.ascontiguousarray(blob.transpose(0, 3, 1, 2), dtype=np.float32)
        return self._fallback.run(blob)


def make_runner(path: str | Path, threads: int = 0, outputs: Optional[list[str]] = None, prefer_openvino: bool = True,
                nhwc_u8: bool = False):
    """nhwc_u8: caller feeds raw HxWx3 uint8 frames (OpenVINO converts on device); the ORT
    fallback converts on the CPU. Check `runner.takes_nhwc_u8` to know which to send."""
    path = str(path)
    dev = openvino_device() if prefer_openvino else None
    if dev:
        try:
            r = FallbackRunner(OpenVINORunner(path, dev, outputs, nhwc_u8=nhwc_u8), path, threads, outputs)
            r.takes_nhwc_u8 = nhwc_u8
            return r
        except Exception as e:  # noqa: BLE001
            log.warning("OpenVINO runner failed for %s (%s); using ONNX Runtime", Path(path).name, e)
    r = ORTRunner(path, threads, outputs)
    r.takes_nhwc_u8 = False
    return r
