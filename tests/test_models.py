"""Model registry: optional models (no download source) never block the model check."""
import urllib.request

import pytest

from videoredact.vision import models as reg


def test_registry_hashes_are_pinned_for_every_downloadable_model():
    for n, spec in reg.MODELS.items():
        if spec.url:
            assert len(spec.sha256) == 64 and int(spec.sha256, 16) >= 0, n
        else:
            assert spec.sha256 == ""


def test_verify_model_detects_a_swapped_file(tmp_path, monkeypatch):
    p = tmp_path / "m.onnx"
    p.write_bytes(b"x" * 1000)
    good = reg.file_sha256(p)
    monkeypatch.setitem(reg.MODELS, "fake", reg.ModelSpec("fake", "m.onnx", "http://x", "MIT", "t", 10, good))
    monkeypatch.setitem(reg.MODELS, "unpinned", reg.ModelSpec("unpinned", "m.onnx", "", "MIT", "t", 10))
    monkeypatch.setattr(reg, "model_path", lambda n: p)
    assert reg.verify_model("fake") == (True, good)
    assert reg.verify_model("unpinned") == (None, good)
    p.write_bytes(b"y" * 1000)
    ok, actual = reg.verify_model("fake")
    assert ok is False and actual != good


def test_ensure_model_rejects_a_download_with_the_wrong_hash(tmp_path, monkeypatch):
    import io
    monkeypatch.setitem(reg.MODELS, "fake", reg.ModelSpec("fake", "m.onnx", "http://x/m.onnx", "MIT", "t", 10, "0" * 64))
    monkeypatch.setattr(reg, "model_path", lambda n: tmp_path / "m.onnx")
    monkeypatch.setattr(reg, "is_available", lambda n: False)
    monkeypatch.setattr("videoredact.paths.settings", {"offline": False})
    monkeypatch.delenv("VIDEOREDACT_OFFLINE", raising=False)      # the package sets it on import

    class Resp(io.BytesIO):
        headers = {"Content-Length": "100"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: Resp(b"z" * 100))
    with pytest.raises(RuntimeError, match="does not match"):
        reg.ensure_model("fake")
    assert not (tmp_path / "m.onnx").exists() and not (tmp_path / "m.part").exists()


def test_optional_models_are_the_ones_without_a_url():
    assert reg.is_optional("yolox_tiny_doc")
    assert not reg.is_optional("yunet")
    assert all(reg.is_optional(n) == (not reg.MODELS[n].url) for n in reg.MODELS)


def test_required_models_skip_absent_optional(monkeypatch):
    monkeypatch.setattr(reg, "is_available", lambda n: n != "yolox_tiny_doc")
    names = reg.required_models()
    assert "yolox_tiny_doc" not in names
    assert "yunet" in names and "yolox_tiny" in names


def test_required_models_include_installed_optional(monkeypatch):
    monkeypatch.setattr(reg, "is_available", lambda n: True)
    assert "yolox_tiny_doc" in reg.required_models()


def test_ensure_model_missing_optional_is_a_clear_error_without_network(monkeypatch):
    # Regression: 0.1.13's Tools > "Check detection models" called ensure_model on every
    # registered model and failed on the optional fine-tuned detector.
    monkeypatch.setattr(reg, "is_available", lambda n: False)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("network call attempted"))
    with pytest.raises(RuntimeError, match="optional"):
        reg.ensure_model("yolox_tiny_doc")
