"""Licensing guard rails (docs/LICENSES.md): nothing AGPL / non-commercial may be shipped."""
from pathlib import Path

ROOT = Path(__file__).parents[1]

FORBIDDEN_IN_SHIPPED = ("ultralytics", "insightface", "boxmot", "strongsort")


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8").lower()


def test_no_agpl_or_noncommercial_dependencies():
    for rel in ("requirements.txt", "pyproject.toml", "installer/videoredact.spec"):
        txt = _read(rel)
        for name in FORBIDDEN_IN_SHIPPED:
            assert name not in txt, f"{name} must not be a shipped dependency ({rel}); see docs/LICENSES.md"


def test_shipped_model_registry_is_permissive():
    from videoredact.vision.models import MODELS
    for name, spec in MODELS.items():
        lic = spec.license.lower()
        assert any(k in lic for k in ("mit", "apache", "bsd")), f"{name}: {spec.license}"
        assert "scrfd" not in name.lower() and "insightface" not in spec.url.lower()


def test_licenses_doc_covers_shipped_models_and_libs():
    doc = _read("docs/LICENSES.md")
    from videoredact.vision.models import MODELS
    for spec in MODELS.values():
        assert spec.filename.lower() in doc, f"{spec.filename} missing from docs/LICENSES.md"
    for pkg in ("pyside6", "onnxruntime", "openvino", "faster-whisper", "trackers", "supervision", "reportlab", "ffmpeg"):
        assert pkg in doc


def test_telemetry_package_excluded_from_installer():
    assert "openvino_telemetry" in _read("installer/videoredact.spec")
    assert "imageio_ffmpeg" in _read("installer/videoredact.spec")   # GPL ffmpeg build in that wheel stays out
