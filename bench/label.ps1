# Starts Label Studio locally (no cloud, no telemetry) serving bench\frames for annotation.
# First run: .\scripts\setup_dev.ps1 then  python -m venv C:\dev\venv-labelstudio; C:\dev\venv-labelstudio\Scripts\pip install label-studio
# Usage:  .\bench\label.ps1 [-VenvPath C:\dev\venv-labelstudio] [-Port 8080]
param(
    [string]$VenvPath = "C:\dev\venv-labelstudio",
    [int]$Port = 8080
)
$repo = Split-Path -Parent $PSScriptRoot
$frames = Join-Path $repo "bench\frames"
$data = Join-Path $repo "bench\labelstudio\data"
New-Item -ItemType Directory -Force $data | Out-Null
if (-not (Test-Path $frames)) { throw "bench\frames not found. Run: python bench\extract_frames.py" }

# Local file serving so tasks can reference bench\frames without uploading anything anywhere.
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = $frames
$env:LABEL_STUDIO_BASE_DATA_DIR = $data
$env:LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK = "false"
$env:COLLECT_ANALYTICS = "false"
$env:LABEL_STUDIO_NO_ANALYTICS = "1"
$env:DISABLE_TELEMETRY = "1"

Write-Host "Label Studio data dir: $data"
Write-Host "Serving frames from:   $frames"
Write-Host "Open http://localhost:$Port  (create an account the first time; it is local only)"
Write-Host "Project setup: Settings > Labeling Interface > Code: paste bench\labeling_config.xml"
Write-Host "Import: bench\labelstudio\tasks.json (from bench\preannotate.py) or add Local Files storage."
& "$VenvPath\Scripts\label-studio.exe" start --port $Port --data-dir $data
