# Builds a new VideoRedact release:
#   1. bumps the version (patch by default; -Bump minor|major|X.Y.Z)        videoredact/__init__.py + pyproject.toml
#   2. PyInstaller one-folder build                                          dist\VideoRedact\
#   3. Inno Setup installer                                                  Published\VideoRedact-<ver>-Setup.exe
#   4. writes Published\latest.json and SHA256SUMS.txt
#   5. commits the bump, tags v<ver>, pushes (unless -NoCommit)
#   6. optionally publishes the GitHub release (-Publish) so installed copies auto-update
#
# Prereqs: scripts\setup_dev.ps1 (venv), `pip install pyinstaller`, Inno Setup 6 (winget install JRSoftware.InnoSetup).
# Usage:   .\installer\build.ps1 [-Bump patch] [-NoModels] [-NoCommit] [-Publish] [-VenvPath C:\dev\venv-videoredact]
param(
    [string]$VenvPath = "C:\dev\venv-videoredact",
    [string]$Bump = "patch",
    [switch]$NoModels,
    [switch]$NoCommit,
    [switch]$Publish,
    [switch]$NoBump
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$py = "$VenvPath\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "venv not found at $VenvPath - run scripts\setup_dev.ps1" }

# ---- 1. version -----------------------------------------------------------
if ($NoBump) { $version = & $py scripts\bump_version.py --print }
else         { $version = & $py scripts\bump_version.py $Bump }
$version = "$version".Trim()
Write-Host "==> Building VideoRedact $version"

# ---- icon + Windows version resource -------------------------------------
if (-not (Test-Path "$repo\installer\videoredact.ico")) { & $py scripts\make_icon.py }
New-Item -ItemType Directory -Force "$repo\build" | Out-Null
$v4 = "$version.0"
$vtuple = ($v4 -split "\.") -join ", "
@"
VSVersionInfo(
  ffi=FixedFileInfo(filevers=($vtuple), prodvers=($vtuple), mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[StringFileInfo([StringTable('040904B0', [
      StringStruct('CompanyName', 'VideoRedact project'),
      StringStruct('FileDescription', 'VideoRedact - video and audio redaction'),
      StringStruct('FileVersion', '$v4'),
      StringStruct('InternalName', 'VideoRedact'),
      StringStruct('LegalCopyright', 'MIT License'),
      StringStruct('OriginalFilename', 'VideoRedact.exe'),
      StringStruct('ProductName', 'VideoRedact'),
      StringStruct('ProductVersion', '$v4')])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])]
)
"@ | Set-Content -Encoding utf8 "$repo\build\version_info.txt"

# ---- FFmpeg ---------------------------------------------------------------
# Ship an LGPL FFmpeg build at bin\ffmpeg.exe (e.g. the "lgpl" variants from
# https://github.com/BtbN/FFmpeg-Builds/releases). For test builds fall back to the imageio-ffmpeg binary.
if (-not (Test-Path "$repo\bin\ffmpeg.exe")) {
    Write-Warning "bin\ffmpeg.exe not found; using the imageio-ffmpeg binary (GPL build) for this build. Replace with an LGPL build before wide distribution."
    New-Item -ItemType Directory -Force "$repo\bin" | Out-Null
    $src = & $py -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
    Copy-Item "$src".Trim() "$repo\bin\ffmpeg.exe"
}

# ---- models (bundled by default so agency machines need no internet) ------
# Detection models (~65 MB) + faster-whisper "small" int8 (~480 MB) go into models\ -> _internal\models.
# Pass -NoModels for a slim download-on-first-use build (remove the models\ folder first).
if (-not $NoModels) {
    $env:VIDEOREDACT_MODELS = "$repo\models"
    & $py scripts\fetch_models.py --whisper small
    Remove-Item Env:\VIDEOREDACT_MODELS
    if (-not (Test-Path "$repo\models\whisper\models--Systran--faster-whisper-small")) { throw "whisper model bundle missing" }
}

# ---- 2. PyInstaller -------------------------------------------------------
if (Test-Path "$repo\dist\VideoRedact") { Remove-Item -Recurse -Force "$repo\dist\VideoRedact" }
& $py -m PyInstaller --noconfirm --clean --distpath "$repo\dist" --workpath "$repo\build\pyi" "$repo\installer\videoredact.spec"
if (-not (Test-Path "$repo\dist\VideoRedact\VideoRedact.exe")) { throw "PyInstaller build failed" }
$size = [math]::Round((Get-ChildItem -Recurse "$repo\dist\VideoRedact" | Measure-Object Length -Sum).Sum / 1MB)
Write-Host "==> Folder build: dist\VideoRedact ($size MB)"

# ---- 3. Inno Setup --------------------------------------------------------
$iscc = @("$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe", "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe", "$env:ProgramFiles\Inno Setup 6\ISCC.exe") |
    Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) { throw "Inno Setup not found. Install with: winget install JRSoftware.InnoSetup" }
New-Item -ItemType Directory -Force "$repo\Published" | Out-Null
& $iscc /Q "/DAppVersion=$version" "$repo\installer\videoredact.iss"
$setup = "$repo\Published\VideoRedact-$version-Setup.exe"
if (-not (Test-Path $setup)) { throw "Installer was not produced" }

# ---- 4. manifest ----------------------------------------------------------
$hash = (Get-FileHash $setup -Algorithm SHA256).Hash.ToLower()
"$hash  VideoRedact-$version-Setup.exe" | Add-Content -Encoding utf8 "$repo\Published\SHA256SUMS.txt"
@{ version = $version; file = "VideoRedact-$version-Setup.exe"; sha256 = $hash; built = (Get-Date).ToString("s");
   size = (Get-Item $setup).Length } | ConvertTo-Json | Set-Content -Encoding utf8 "$repo\Published\latest.json"
Write-Host "==> Installer: $setup ($([math]::Round((Get-Item $setup).Length/1MB)) MB)"

# ---- 5. commit + tag ------------------------------------------------------
if (-not $NoCommit) {
    git add videoredact/__init__.py pyproject.toml
    git commit -q -m "Release $version" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
    git tag -a "v$version" -m "VideoRedact $version"
    git push -q origin HEAD
    git push -q origin "v$version"
    Write-Host "==> Committed and tagged v$version"
}

# ---- 6. publish -----------------------------------------------------------
if ($Publish) {
    & "$repo\scripts\publish_release.ps1" -Version $version
}
Write-Host "==> Done."
