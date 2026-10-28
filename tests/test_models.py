"""Model registry: optional models (no download source) never block the model check."""
import urllib.request

import pytest

from videoredact.vision import models as reg


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
