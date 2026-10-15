"""Bump the version and keep every copy in sync:
  videoredact/__init__.py   (__version__, the runtime source of truth: works in frozen builds)
  VERSION                   (plain text, for scripts / CI / humans)
  pyproject.toml            (package metadata)
  CHANGELOG.md              ("## [Unreleased]" section becomes "## [X.Y.Z] - YYYY-MM-DD")

Usage:  python scripts/bump_version.py [patch|minor|major|X.Y.Z] [--print] [--notes [X.Y.Z]] [--check]
  default      bump patch, print the new version
  --print      print the current version and exit
  --notes      print the CHANGELOG section for the given (default: current) version and exit
  --check      exit 1 if the copies disagree (used by tests/CI)
"""
import datetime as _dt
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INIT = ROOT / "videoredact" / "__init__.py"
PYPROJECT = ROOT / "pyproject.toml"
VERSION_FILE = ROOT / "VERSION"
CHANGELOG = ROOT / "CHANGELOG.md"


def current() -> str:
    return re.search(r'__version__\s*=\s*"(\d+\.\d+\.\d+)"', INIT.read_text(encoding="utf-8")).group(1)


def bump(kind: str) -> str:
    major, minor, patch = (int(x) for x in current().split("."))
    if re.fullmatch(r"\d+\.\d+\.\d+", kind):
        return kind
    if kind == "major":
        return f"{major + 1}.0.0"
    if kind == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def write(version: str) -> None:
    s = INIT.read_text(encoding="utf-8")
    s = re.sub(r'__version__\s*=\s*"[^"]+"', f'__version__ = "{version}"', s)
    INIT.write_text(s, encoding="utf-8", newline="\n")
    p = PYPROJECT.read_text(encoding="utf-8")
    p = re.sub(r'(?m)^version\s*=\s*"[^"]+"', f'version = "{version}"', p, count=1)
    PYPROJECT.write_text(p, encoding="utf-8", newline="\n")
    VERSION_FILE.write_text(version + "\n", encoding="utf-8", newline="\n")
    promote_changelog(version)


def promote_changelog(version: str) -> None:
    """Turn '## [Unreleased]' into the released section (dated today) and open a fresh one."""
    if not CHANGELOG.exists():
        return
    s = CHANGELOG.read_text(encoding="utf-8")
    if f"## [{version}]" in s:
        return
    today = _dt.date.today().isoformat()
    if "## [Unreleased]" not in s:
        s = s.replace("\n## [", f"\n## [Unreleased]\n\n## [", 1)
    s = s.replace("## [Unreleased]", f"## [Unreleased]\n\n## [{version}] - {today}", 1)
    CHANGELOG.write_text(s, encoding="utf-8", newline="\n")


def notes(version: str) -> str:
    """The CHANGELOG body for one version (used as the GitHub release notes)."""
    if not CHANGELOG.exists():
        return ""
    s = CHANGELOG.read_text(encoding="utf-8")
    m = re.search(rf"(?ms)^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)", s)
    return (m.group(1).strip() if m else "")


def check() -> list[str]:
    """Names of the copies that disagree with videoredact/__init__.py."""
    v = current()
    bad = []
    if VERSION_FILE.read_text(encoding="utf-8").strip() != v:
        bad.append("VERSION")
    if not re.search(rf'(?m)^version\s*=\s*"{re.escape(v)}"', PYPROJECT.read_text(encoding="utf-8")):
        bad.append("pyproject.toml")
    if CHANGELOG.exists() and f"## [{v}]" not in CHANGELOG.read_text(encoding="utf-8"):
        bad.append("CHANGELOG.md")
    return bad


if __name__ == "__main__":
    # The changelog contains arrows/dashes outside cp1252; PowerShell's default console
    # encoding would raise UnicodeEncodeError when printing the release notes.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--print" in sys.argv:
        print(current())
        sys.exit(0)
    if "--notes" in sys.argv:
        print(notes(args[0] if args else current()))
        sys.exit(0)
    if "--check" in sys.argv:
        bad = check()
        print("version copies in sync" if not bad else "OUT OF SYNC: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    new = bump(args[0] if args else "patch")
    write(new)
    print(new)
