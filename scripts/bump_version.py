"""Bump the single source of truth for the version (videoredact/__init__.py)
and mirror it into pyproject.toml.

Usage:  python scripts/bump_version.py [patch|minor|major|X.Y.Z] [--print]
Default: patch. Prints the new version on stdout.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INIT = ROOT / "videoredact" / "__init__.py"
PYPROJECT = ROOT / "pyproject.toml"


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


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--print" in sys.argv:
        print(current())
        sys.exit(0)
    new = bump(args[0] if args else "patch")
    write(new)
    print(new)
