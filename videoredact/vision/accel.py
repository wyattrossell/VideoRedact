"""Hardware detection and acceleration-tier selection.

Tiers (auto-selected, highest first; each falls back to the next when the
runtime for it is not installed or fails to create a session):

  nvidia  - NVIDIA GPU with a CUDA-capable driver. Target runtime:
            onnxruntime-gpu (CUDAExecutionProvider / TensorrtExecutionProvider).
  intel   - Intel integrated/discrete GPU or NPU. Target runtime:
            onnxruntime-openvino (OpenVINOExecutionProvider, device GPU/NPU/CPU).
  directml- any other DirectX 12 GPU on Windows (AMD iGPU): onnxruntime-directml.
  cpu     - ONNX Runtime CPU provider (always available, current default).

Phase 0 ships detection + reporting only; the ORT session factory in
detector.py consults `preferred_providers()` and silently keeps CPU when the
accelerated providers are not present, so nothing changes for users until
the accelerated wheels are bundled (Phase 1).
"""
from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from typing import Optional

_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


@dataclass
class HardwareInfo:
    os: str = ""
    cpu: str = ""
    cores: int = 0
    logical: int = 0
    ram_gb: float = 0.0
    gpus: list[str] = field(default_factory=list)        # all display adapters
    nvidia_gpu: str = ""
    nvidia_driver: str = ""
    cuda_version: str = ""                               # from nvidia-smi (driver's max CUDA)
    nvidia_vram_mb: int = 0
    intel_gpu: str = ""
    intel_npu: str = ""
    ort_providers: list[str] = field(default_factory=list)
    tier: str = "cpu"                                     # nvidia | intel | directml | cpu
    active_provider: str = "CPUExecutionProvider"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        acc = {"nvidia": f"NVIDIA {self.nvidia_gpu} (CUDA {self.cuda_version})",
               "intel": f"Intel {self.intel_gpu or self.intel_npu}",
               "directml": f"DirectML ({self.gpus[0] if self.gpus else 'GPU'})",
               "cpu": "CPU only"}[self.tier]
        return (f"{self.cpu} ({self.cores}c/{self.logical}t), {self.ram_gb:.0f} GB, {self.os}; "
                f"tier={self.tier} via {self.active_provider} [{acc}]")


def _run(cmd: list[str], timeout: float = 8.0) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, creationflags=_CREATE_NO_WINDOW)
        return r.stdout if r.returncode == 0 else ""
    except Exception:
        return ""


def _powershell(script: str) -> str:
    if sys.platform != "win32":
        return ""
    return _run(["powershell", "-NoProfile", "-Command", script], timeout=20)


@lru_cache(maxsize=1)
def detect_hardware() -> HardwareInfo:
    hw = HardwareInfo(os=f"{platform.system()} {platform.release()} ({platform.version()})",
                      cpu=platform.processor() or "", logical=os.cpu_count() or 0)
    # ---- CPU / RAM / adapters (Windows CIM; psutil-free) ----
    if sys.platform == "win32":
        out = _powershell(
            "$c=Get-CimInstance Win32_Processor | Select-Object -First 1; "
            "Write-Output ('CPU|' + $c.Name + '|' + $c.NumberOfCores + '|' + $c.NumberOfLogicalProcessors); "
            "Write-Output ('RAM|' + [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)); "
            "Get-CimInstance Win32_VideoController | ForEach-Object { Write-Output ('GPU|' + $_.Name + '|' + $_.DriverVersion) }; "
            "Get-PnpDevice -PresentOnly -ErrorAction SilentlyContinue | Where-Object { $_.FriendlyName -match 'NPU|Neural|AI Boost' } "
            "| ForEach-Object { Write-Output ('NPU|' + $_.FriendlyName) }")
        for line in out.splitlines():
            parts = line.strip().split("|")
            if parts[0] == "CPU" and len(parts) >= 4:
                hw.cpu = parts[1].strip()
                hw.cores = int(parts[2] or 0)
                hw.logical = int(parts[3] or hw.logical)
            elif parts[0] == "RAM" and len(parts) >= 2:
                try:
                    hw.ram_gb = float(parts[1])
                except ValueError:
                    pass
            elif parts[0] == "GPU" and len(parts) >= 2:
                name = parts[1].strip()
                hw.gpus.append(name)
                low = name.lower()
                if "nvidia" in low and not hw.nvidia_gpu:
                    hw.nvidia_gpu = name
                    hw.nvidia_driver = parts[2].strip() if len(parts) > 2 else ""
                elif "intel" in low and not hw.intel_gpu:
                    hw.intel_gpu = name
            elif parts[0] == "NPU" and len(parts) >= 2:
                hw.intel_npu = parts[1].strip()
    # ---- NVIDIA details ----
    smi = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    if smi.strip():
        first = smi.strip().splitlines()[0].split(",")
        if len(first) >= 3:
            hw.nvidia_gpu = hw.nvidia_gpu or first[0].strip()
            hw.nvidia_driver = first[1].strip()
            m = re.search(r"(\d+)", first[2])
            hw.nvidia_vram_mb = int(m.group(1)) if m else 0
        m = re.search(r"CUDA Version:\s*([\d.]+)", _run(["nvidia-smi"]))
        if m:
            hw.cuda_version = m.group(1)
    # ---- ONNX Runtime providers ----
    try:
        import onnxruntime as ort
        hw.ort_providers = list(ort.get_available_providers())
    except Exception as e:  # noqa: BLE001
        hw.notes.append(f"onnxruntime unavailable: {e}")
    hw.tier, hw.active_provider = _choose_tier(hw)
    return hw


def _choose_tier(hw: HardwareInfo) -> tuple[str, str]:
    """Highest tier whose runtime is actually installed; notes explain fallbacks."""
    p = set(hw.ort_providers)
    if hw.nvidia_gpu:
        if "TensorrtExecutionProvider" in p:
            return "nvidia", "TensorrtExecutionProvider"
        if "CUDAExecutionProvider" in p:
            return "nvidia", "CUDAExecutionProvider"
        hw.notes.append("NVIDIA GPU present but onnxruntime-gpu is not installed; using CPU")
    if hw.intel_gpu or hw.intel_npu:
        try:
            from .runner import openvino_device
            dev = openvino_device()
        except Exception:  # noqa: BLE001
            dev = None
        if dev:
            return "intel", f"openvino:{dev}"
        hw.notes.append("Intel GPU/NPU present but the `openvino` package is not installed; using CPU")
    if sys.platform == "win32" and hw.gpus and "DmlExecutionProvider" in p:
        return "directml", "DmlExecutionProvider"
    return "cpu", "CPUExecutionProvider"


def preferred_providers() -> list:
    """Provider list for ort.InferenceSession, best tier first, CPU last.
    Entries may be (name, options) tuples."""
    hw = detect_hardware()
    chain: list = []
    if hw.tier == "nvidia":
        if hw.active_provider == "TensorrtExecutionProvider":
            chain.append(("TensorrtExecutionProvider", {"trt_fp16_enable": True}))
        chain.append("CUDAExecutionProvider")
    elif hw.tier == "intel":
        pass   # the OpenVINO Python API runner handles this tier (vision/runner.py); ORT stays CPU
    elif hw.tier == "directml":
        chain.append("DmlExecutionProvider")
    chain.append("CPUExecutionProvider")
    return chain


def accel_label() -> str:
    hw = detect_hardware()
    return {"nvidia": "NVIDIA GPU", "intel": "Intel GPU/NPU", "directml": "DirectML GPU", "cpu": "CPU"}[hw.tier]
