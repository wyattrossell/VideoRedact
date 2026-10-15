"""Update notification via GitHub Releases (notify only; no automatic installs).

The build pipeline (installer/build.ps1 + scripts/publish_release.ps1) tags every build
vX.Y.Z and uploads `VideoRedact-X.Y.Z-Setup.exe` as a release asset. A few seconds after
startup (if `check_updates` is on) and from Help > Check for updates, the app asks the GitHub
API for the latest release in a background thread with a short timeout. If it is newer than
the running version the user is told and offered a link to the release page. Nothing is
downloaded or installed by the application; the operator (or IT) runs the installer.

Only the public GitHub API is contacted, with no credentials and no case data. Machines
without internet simply get no notification (the failure is silent unless the check was
started manually). See docs/CJIS_COMPLIANCE.md section 2.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.request
import webbrowser
from dataclasses import dataclass
from typing import Optional

from videoredact import __version__

REPO = "wyattrossell/VideoRedact"
API_LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{REPO}/releases"
ASSET_RE = re.compile(r"^VideoRedact-(\d+\.\d+\.\d+)-Setup\.exe$", re.I)
DEFAULT_TIMEOUT_S = 5.0

log = logging.getLogger(__name__)


@dataclass
class UpdateInfo:
    version: str
    url: str          # installer asset download URL ("" if the release has no installer yet)
    size: int
    notes: str
    page: str         # release page to open in the browser


def parse_version(v: str) -> tuple[int, ...]:
    """'v1.2.3' / '1.2' / '2.0.0-beta' -> (major, minor, patch); non-numeric parts count as 0."""
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


def parse_release(data: dict) -> Optional[UpdateInfo]:
    """Build UpdateInfo from a GitHub 'release' JSON object (pure; used by tests)."""
    tag = data.get("tag_name") or data.get("name") or ""
    page = data.get("html_url") or RELEASES_PAGE
    notes = (data.get("body") or "").strip()
    for a in data.get("assets", []):
        m = ASSET_RE.match(a.get("name", ""))
        if m:
            return UpdateInfo(version=m.group(1), url=a["browser_download_url"], size=int(a.get("size", 0)),
                              notes=notes, page=page)
    if tag:
        return UpdateInfo(version=tag.lstrip("vV"), url="", size=0, notes=notes, page=page)
    return None


def fetch_latest(timeout: float = DEFAULT_TIMEOUT_S) -> Optional[UpdateInfo]:
    """Latest release on GitHub, or None if there is none. Raises on network errors
    (callers running at startup swallow the error; the manual check shows it)."""
    req = urllib.request.Request(API_LATEST, headers={
        "User-Agent": f"VideoRedact/{__version__}", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    return parse_release(data)


def check_for_update(timeout: float = DEFAULT_TIMEOUT_S) -> Optional[UpdateInfo]:
    """Latest release if it is newer than the running version, else None. Never raises:
    offline machines get None (the reason is logged at debug level)."""
    try:
        info = fetch_latest(timeout)
    except Exception as e:  # noqa: BLE001  no network, proxy, DNS, rate limit...
        log.debug("update check failed: %s", e)
        return None
    if info and is_newer(info.version):
        return info
    return None


def open_release_page(info: Optional[UpdateInfo] = None) -> bool:
    """Open the release (or releases list) in the default browser. Returns False if that failed."""
    url = (info.page if info and info.page else RELEASES_PAGE)
    try:
        return bool(webbrowser.open(url, new=2))
    except Exception:  # noqa: BLE001
        return False
