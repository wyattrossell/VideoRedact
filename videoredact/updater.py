"""Self-update via GitHub Releases.

The build pipeline (installer/build.ps1 + scripts/publish_release.ps1) tags
every build vX.Y.Z and uploads `VideoRedact-X.Y.Z-Setup.exe` as a release
asset. At startup (and from Help > Check for updates) the app asks the GitHub
API for the latest release; if it is newer than the running version it
offers to download the installer and run it. The installer closes the app,
upgrades in place (same AppId) and restarts it.

Only the public GitHub API is contacted, with no credentials. Machines
without internet simply get a "could not check" result.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from videoredact import __version__

REPO = "wyattrossell/VideoRedact"
API_LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{REPO}/releases"
ASSET_RE = re.compile(r"^VideoRedact-(\d+\.\d+\.\d+)-Setup\.exe$", re.I)
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


@dataclass
class UpdateInfo:
    version: str
    url: str          # installer asset download URL
    size: int
    notes: str
    page: str


def parse_version(v: str) -> tuple[int, ...]:
    v = v.strip().lstrip("vV")
    parts = []
    for p in v.split(".")[:3]:
        m = re.match(r"\d+", p)
        parts.append(int(m.group()) if m else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def is_newer(candidate: str, current: str = __version__) -> bool:
    return parse_version(candidate) > parse_version(current)


def fetch_latest(timeout: float = 8.0) -> Optional[UpdateInfo]:
    """Return info about the latest release, or None if there is no release
    with an installer asset. Raises on network errors."""
    req = urllib.request.Request(API_LATEST, headers={
        "User-Agent": f"VideoRedact/{__version__}", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    tag = data.get("tag_name") or data.get("name") or ""
    for a in data.get("assets", []):
        m = ASSET_RE.match(a.get("name", ""))
        if m:
            return UpdateInfo(version=m.group(1), url=a["browser_download_url"], size=int(a.get("size", 0)),
                              notes=(data.get("body") or "").strip(), page=data.get("html_url") or RELEASES_PAGE)
    if tag:
        return UpdateInfo(version=tag.lstrip("v"), url="", size=0, notes=(data.get("body") or "").strip(),
                          page=data.get("html_url") or RELEASES_PAGE)
    return None


def check_for_update(timeout: float = 8.0) -> Optional[UpdateInfo]:
    """Latest release if it is newer than the running version, else None."""
    info = fetch_latest(timeout)
    if info and is_newer(info.version):
        return info
    return None


def download(info: UpdateInfo, progress: Optional[Callable[[float, str], None]] = None,
             cancel: Optional[Callable[[], bool]] = None) -> Path:
    if not info.url:
        raise RuntimeError("This release has no installer attached. Download it from " + info.page)
    dest = Path(tempfile.gettempdir()) / f"VideoRedact-{info.version}-Setup.exe"
    tmp = dest.with_suffix(".part")
    req = urllib.request.Request(info.url, headers={"User-Agent": f"VideoRedact/{__version__}"})
    with urllib.request.urlopen(req, timeout=30) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or info.size or 0)
        done = 0
        while True:
            if cancel and cancel():
                raise RuntimeError("cancelled")
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if progress and total:
                progress(done / total, f"Downloading update {done / 1e6:.0f} / {total / 1e6:.0f} MB")
    if total and tmp.stat().st_size != total:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("Download incomplete; please try again.")
    tmp.replace(dest)
    return dest


def installer_args(setup_path: str, exe_path: str) -> list[str]:
    """Inno Setup arguments for an in-app upgrade.

    /NOCLOSEAPPLICATIONS: the Restart Manager check hung for 20+ minutes on a
    test machine even with no VideoRedact process alive, so we skip it; the
    app quits itself before the installer starts. /ALLUSERS vs /CURRENTUSER
    follows where the running copy is installed so a per-machine install does
    not silently gain a second per-user copy."""
    exe_lower = exe_path.lower()
    per_machine = any(k in exe_lower for k in ("\program files", "\programfiles"))
    scope = "/ALLUSERS" if per_machine else "/CURRENTUSER"
    return [setup_path, "/SILENT", "/SUPPRESSMSGBOXES", "/NOCLOSEAPPLICATIONS", "/NORESTART", scope]


def launch_installer(path: Path, restart: bool = True) -> None:
    """Start the installer detached; relaunch the app when it finishes.
    The caller should quit the app right after calling this."""
    exe = sys.executable if getattr(sys, "frozen", False) else ""
    args = installer_args(str(path), exe)
    if sys.platform == "win32":
        quoted = " ".join(f'"{a}"' if " " in a else a for a in args)
        cmd = f'{quoted}'
        if restart and exe:
            cmd += f' && start "" "{exe}"'
        flags = _CREATE_NO_WINDOW | getattr(subprocess, "DETACHED_PROCESS", 0)
        subprocess.Popen(["cmd", "/c", cmd], creationflags=flags, close_fds=True)
    else:
        subprocess.Popen(args, close_fds=True)
