"""Pre-download all detection models (for offline machines / installer bundling).

Usage:  python scripts/fetch_models.py [--whisper small]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from videoredact.vision.models import MODELS, ensure_model, is_available, model_path  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--whisper", default="", help="also download this faster-whisper model size (e.g. small)")
    ap.add_argument("--only", nargs="*", default=None, help="subset of model names")
    args = ap.parse_args()
    names = args.only or list(MODELS)
    for name in names:
        if is_available(name):
            print(f"[ok]      {name:12s} {model_path(name)}")
            continue
        print(f"[fetch]   {name:12s} {MODELS[name].url}")
        ensure_model(name, lambda p, m: print(f"\r          {m}", end=""))
        print(f"\n[ok]      {name:12s} {model_path(name)}")
    if args.whisper:
        from videoredact.audio.transcribe import Transcriber
        print(f"[fetch]   whisper-{args.whisper}")
        Transcriber(args.whisper)
        print("[ok]      whisper model ready")


if __name__ == "__main__":
    main()
