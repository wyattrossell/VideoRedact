# Publishes Published\VideoRedact-<version>-Setup.exe as a GitHub release (tag v<version>).
# The app's auto-updater reads the "latest" release and expects exactly that asset name.
#
# Auth: uses GitHub CLI (`gh`) when installed and logged in; otherwise uses the token stored
# by Git Credential Manager for github.com (the same credential `git push` uses), or $env:GITHUB_TOKEN.
#
# Usage: .\scripts\publish_release.ps1 [-Version 0.1.3] [-Notes "text"] [-Draft]
param(
    [string]$Version,
    [string]$Notes,
    [switch]$Draft
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$slug = "wyattrossell/VideoRedact"
Set-Location $repo

if (-not $Version) {
    $Version = (Get-Content "$repo\videoredact\__init__.py" | Select-String '__version__\s*=\s*"([^"]+)"').Matches[0].Groups[1].Value
}
$asset = "$repo\Published\VideoRedact-$Version-Setup.exe"
if (-not (Test-Path $asset)) { throw "Installer not found: $asset  (run installer\build.ps1 first)" }
$tag = "v$Version"
if (-not $Notes) {
    $Notes = "VideoRedact $Version`n`nInstaller for Windows 10/11 (64-bit). Existing installations update automatically from Help > Check for updates or at startup."
}

# Make sure the tag exists on GitHub
$remoteTag = git ls-remote --tags origin "refs/tags/$tag"
if (-not $remoteTag) {
    if (-not (git tag -l $tag)) { git tag -a $tag -m "VideoRedact $Version" }
    git push origin $tag
}

$gh = Get-Command gh -ErrorAction SilentlyContinue
if ($gh) {
    $flags = @()
    if ($Draft) { $flags += "--draft" }
    gh release create $tag $asset --repo $slug --title "VideoRedact $Version" --notes $Notes @flags
    Write-Host "Release $tag published via gh."
    exit 0
}

# --- REST fallback ---------------------------------------------------------
$token = $env:GITHUB_TOKEN
if (-not $token) {
    $cred = "protocol=https`nhost=github.com`n`n" | git credential fill
    $token = ($cred | Select-String '^password=(.*)$').Matches[0].Groups[1].Value
}
if (-not $token) { throw "No GitHub credential found. Install GitHub CLI (winget install GitHub.cli; gh auth login) or set GITHUB_TOKEN." }
$headers = @{ Authorization = "Bearer $token"; Accept = "application/vnd.github+json"; "User-Agent" = "VideoRedact-publish" }

$existing = $null
try { $existing = Invoke-RestMethod -Headers $headers -Uri "https://api.github.com/repos/$slug/releases/tags/$tag" } catch {}
if (-not $existing) {
    $body = @{ tag_name = $tag; name = "VideoRedact $Version"; body = $Notes; draft = [bool]$Draft; prerelease = $false } | ConvertTo-Json
    $existing = Invoke-RestMethod -Method Post -Headers $headers -Uri "https://api.github.com/repos/$slug/releases" -Body $body -ContentType "application/json"
    Write-Host "Created release $tag"
}
# remove an asset with the same name if re-publishing
foreach ($a in $existing.assets) {
    if ($a.name -eq (Split-Path -Leaf $asset)) {
        Invoke-RestMethod -Method Delete -Headers $headers -Uri $a.url | Out-Null
    }
}
$uploadUrl = ($existing.upload_url -replace "\{\?name,label\}", "") + "?name=" + [uri]::EscapeDataString((Split-Path -Leaf $asset))
Write-Host "Uploading $(Split-Path -Leaf $asset) ($([math]::Round((Get-Item $asset).Length/1MB)) MB)..."
$resp = Invoke-RestMethod -Method Post -Headers $headers -Uri $uploadUrl -InFile $asset -ContentType "application/octet-stream"
Write-Host "Published: $($resp.browser_download_url)"
