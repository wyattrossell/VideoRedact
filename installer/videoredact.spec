# PyInstaller spec for VideoRedact (run via installer/build.ps1).
# One-folder build: dist/VideoRedact/VideoRedact.exe + _internal/.
import re
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = Path(SPECPATH).parent
VERSION = re.search(r'__version__\s*=\s*"([^"]+)"', (ROOT / "videoredact" / "__init__.py").read_text()).group(1)

datas, binaries, hiddenimports = [], [], []
for pkg in ("onnxruntime", "ctranslate2", "faster_whisper", "soundfile", "tokenizers", "huggingface_hub"):
    try:
        d, b, h = collect_all(pkg)
    except Exception:
        continue
    datas += d
    binaries += b
    hiddenimports += h
hiddenimports += collect_submodules("videoredact")
hiddenimports += ["PySide6.QtMultimedia", "reportlab.graphics.barcode", "reportlab.graphics.barcode.common"]

# Bundled FFmpeg (bin/) and optional pre-fetched models (models/) land in _internal/
if (ROOT / "bin").exists():
    datas.append((str(ROOT / "bin"), "bin"))
if (ROOT / "models").exists() and any(p for p in (ROOT / "models").rglob("*") if p.is_file()):
    datas.append((str(ROOT / "models"), "models"))

version_file = ROOT / "build" / "version_info.txt"
icon = ROOT / "installer" / "videoredact.ico"

a = Analysis(
    [str(ROOT / "videoredact" / "app.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    # imageio_ffmpeg is a dev-only fallback; the installer ships bin/ffmpeg.exe instead
    excludes=["imageio_ffmpeg", "torch", "tensorflow", "matplotlib", "tkinter", "PyQt5", "PyQt6",
              "IPython", "jupyter", "pytest", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets",
              "PySide6.Qt3DCore", "PySide6.QtQuick", "PySide6.QtQml", "PySide6.QtCharts", "PySide6.QtDataVisualization",
              "PySide6.QtPdf", "PySide6.QtBluetooth", "PySide6.QtNfc", "PySide6.QtSensors", "PySide6.QtSerialPort",
              "PySide6.QtRemoteObjects", "PySide6.QtScxml", "PySide6.QtTest", "PySide6.QtDesigner", "PySide6.QtHelp"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="VideoRedact",
    icon=str(icon) if icon.exists() else None,
    version=str(version_file) if version_file.exists() else None,
    console=False,
    disable_windowed_traceback=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="VideoRedact")
