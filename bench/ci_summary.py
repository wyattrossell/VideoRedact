"""Print (and append to $GITHUB_STEP_SUMMARY) a short table for a bench/run_bench.py result.

Usage: python bench/ci_summary.py bench/results/ci_v0.1.11.json [title]
"""
import json
import os
import sys


def main() -> None:
    path = sys.argv[1]
    title = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(path)
    d = json.load(open(path, encoding="utf-8"))
    a = d.get("aggregate", {})
    lines = [f"## Benchmark {title}", "", f"VideoRedact {d.get('version')} on {d.get('hardware')}", "",
             "| metric | value |", "|---|---|"]
    for k in ("clips", "frames", "autodetect_fps", "autodetect_parallel_fps", "export_fps"):
        if k in a:
            lines.append(f"| {k} | {a[k]} |")
    acc = a.get("accuracy") or {}
    for k in ("recall", "precision", "leak_rate", "labelled_frames"):
        if k in acc:
            lines.append(f"| {k} | {acc[k]} |")
    text = "\n".join(lines)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
