from videoredact.vision import accel


def test_detect_hardware_runs_and_falls_back_sanely():
    hw = accel.detect_hardware()
    assert hw.tier in ("nvidia", "intel", "directml", "cpu")
    assert hw.active_provider in hw.ort_providers or hw.active_provider == "CPUExecutionProvider"
    provs = accel.preferred_providers()
    assert provs[-1] == "CPUExecutionProvider"
    assert hw.summary()


def test_choose_tier_prefers_installed_runtime():
    hw = accel.HardwareInfo(nvidia_gpu="RTX", intel_gpu="UHD", ort_providers=["CPUExecutionProvider"])
    assert accel._choose_tier(hw) == ("cpu", "CPUExecutionProvider")
    assert any("onnxruntime-gpu" in n for n in hw.notes)
    hw2 = accel.HardwareInfo(nvidia_gpu="RTX", ort_providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert accel._choose_tier(hw2) == ("nvidia", "CUDAExecutionProvider")
    hw3 = accel.HardwareInfo(intel_npu="Intel AI Boost", ort_providers=["OpenVINOExecutionProvider", "CPUExecutionProvider"])
    assert accel._choose_tier(hw3) == ("intel", "OpenVINOExecutionProvider")
