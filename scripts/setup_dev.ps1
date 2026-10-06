# Creates the development virtual environment at a short local path and installs requirements.
# Usage:  .\scripts\setup_dev.ps1 [-VenvPath C:\dev\venv-videoredact] [-Python python]
param(
    [string]$VenvPath = "C:\dev\venv-videoredact",
    [string]$Python = "python"
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot

& $Python --version
if (-not (Test-Path (Split-Path -Parent $VenvPath))) { New-Item -ItemType Directory -Force (Split-Path -Parent $VenvPath) | Out-Null }
if (-not (Test-Path "$VenvPath\Scripts\python.exe")) {
    Write-Host "Creating venv at $VenvPath"
    & $Python -m venv $VenvPath
}
$py = "$VenvPath\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install -r "$repo\requirements.txt"
Write-Host ""
Write-Host "Done. Run the app with:  $py -m videoredact   (from $repo)"
Write-Host "Optional: pre-download models:  $py scripts\fetch_models.py --whisper small"
