# Launches VideoRedact from source using the dev venv.
param([string]$VenvPath = "C:\dev\venv-videoredact")
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
& "$VenvPath\Scripts\python.exe" -m videoredact @args
