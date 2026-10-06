# PyInstaller spec for VideoRedact (run via installer/build.ps1).
# Produces dist/VideoRedact/ (one-folder build: faster start, easier to debug than one-file).
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent

datas, binaries, hiddenimports = [], [], []
for pkg in ("onnxruntime", "ctranslate2", "faster_whisper", "imageio_ffmpeg", "soundfile"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h
hiddenimports += collect_submodules("videoredact")
hiddenimports += ["PySide6.QtMultimedia", "scipy.signal", "scipy.special", "reportlab.graphics.barcode"]

# Bundled FFmpeg (LGPL build) and pre-fetched models, if present
if (ROOT / "bin").exists():
    datas.append((str(ROOT / "bin"), "bin"))
if (ROOT / "models").exists() and any((ROOT / "models").iterdir()):
    datas.append((str(ROOT / "models"), "models"))

a = Analysis(
    [str(ROOT / "videoredact" / "app.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    excludes=["torch", "tensorflow", "matplotlib", "tkinter", "PyQt5", "PyQt6"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="VideoRedact",
    icon=str(ROOT / "installer" / "videoredact.ico") if (ROOT / "installer" / "videoredact.ico").exists() else None,
    console=False,
    disable_windowed_traceback=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="VideoRedact")
