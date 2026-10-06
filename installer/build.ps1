# Builds the Windows distribution: PyInstaller folder build + Inno Setup installer.
# Prereqs: dev venv (scripts\setup_dev.ps1), `pip install pyinstaller`, Inno Setup 6
# (https://jrsoftware.org/isinfo.php) for the installer step.
# Usage: .\installer\build.ps1 [-VenvPath C:\dev\venv-videoredact] [-SkipInstaller] [-BundleModels]
param(
    [string]$VenvPath = "C:\dev\venv-videoredact",
    [switch]$SkipInstaller,
    [switch]$BundleModels
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$py = "$VenvPath\Scripts\python.exe"

# 1. FFmpeg: expect an LGPL build at bin\ffmpeg.exe (e.g. BtbN "gpl" builds are NOT ok; use the
#    "lgpl" variants from https://github.com/BtbN/FFmpeg-Builds/releases). Fall back to imageio's binary for testing.
if (-not (Test-Path "$repo\bin\ffmpeg.exe")) {
    Write-Warning "bin\ffmpeg.exe not found; copying the imageio-ffmpeg binary for a TEST build (license: check before distributing)."
    New-Item -ItemType Directory -Force "$repo\bin" | Out-Null
    $src = & $py -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
    Copy-Item $src "$repo\bin\ffmpeg.exe"
}

# 2. Optional: bundle detection + whisper models so the installer works offline
if ($BundleModels) {
    $env:VIDEOREDACT_MODELS = "$repo\models"
    & $py scripts\fetch_models.py --whisper small
    Remove-Item Env:\VIDEOREDACT_MODELS
}

# 3. PyInstaller
& $py -m pip install pyinstaller | Out-Null
if (Test-Path "$repo\build") { Remove-Item -Recurse -Force "$repo\build" }
if (Test-Path "$repo\dist\VideoRedact") { Remove-Item -Recurse -Force "$repo\dist\VideoRedact" }
& $py -m PyInstaller --noconfirm --clean "$repo\installer\videoredact.spec"
if (-not (Test-Path "$repo\dist\VideoRedact\VideoRedact.exe")) { throw "PyInstaller build failed" }
Write-Host "Folder build at dist\VideoRedact"

# 4. Inno Setup
if (-not $SkipInstaller) {
    $iscc = @("${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe", "$env:ProgramFiles\Inno Setup 6\ISCC.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $iscc) { Write-Warning "Inno Setup not found; skipping installer. Install from https://jrsoftware.org/isdl.php"; exit 0 }
    $version = & $py -c "import videoredact; print(videoredact.__version__)"
    & $iscc "/DAppVersion=$version" "$repo\installer\videoredact.iss"
    Write-Host "Installer written to installer\Output"
}
