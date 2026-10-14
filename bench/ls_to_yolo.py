"""Convert a Label Studio JSON export (annotations, not predictions) to the YOLO
per-frame label files the benchmark reads.

Usage:  python bench/ls_to_yolo.py export.json --out bench/labels
Writes bench/labels/classes.txt and bench/labels/<clip_stem>/<clip_stem>_f000123.txt
(one line per box: `<class_id> cx cy w h`, normalized). Frames that were reviewed
but contain nothing get an empty .txt so the benchmark counts them as labelled.
Label Studio's "Export -> JSON" format is expected (list of tasks with `annotations`).
"""
import argparse
import json
import re
from pathlib import Path
from urllib.parse import unquote

CLASSES = ["face", "screen", "phone_screen", "document", "plate", "id_card"]


def task_image_path(task: dict) -> str:
    img = task.get("data", {}).get("image", "")
    img = unquote(img)
    m = re.search(r"d=(.+)$", img)
    if m:
        img = m.group(1)
    return img.replace("\\", "/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_json")
    ap.add_argument("--out", default="bench/labels")
    a = ap.parse_args()
    tasks = json.loads(Path(a.export_json).read_text(encoding="utf-8"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    n_files = n_boxes = 0
    for task in tasks:
        anns = [x for x in task.get("annotations", []) if not x.get("was_cancelled")]
        if not anns:
            continue   # not reviewed -> not labelled
        img = task_image_path(task)
        stem = Path(img).stem                      # <clip_stem>_f000123
        if "_f" not in stem:
            continue
        clip_stem = stem.rsplit("_f", 1)[0]
        lines = []
        for r in anns[-1].get("result", []):       # latest annotation wins
            if r.get("type") != "rectanglelabels":
                continue
            v = r["value"]
            labels = v.get("rectanglelabels", [])
            if not labels:
                continue
            cls = labels[0]
            if cls not in CLASSES:
                CLASSES.append(cls)
            x, y, w, h = v["x"] / 100, v["y"] / 100, v["width"] / 100, v["height"] / 100
            lines.append(f"{CLASSES.index(cls)} {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}")
            n_boxes += 1
        d = out / clip_stem
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        n_files += 1
    (out / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    print(f"wrote {n_files} label files, {n_boxes} boxes, classes={CLASSES} -> {out}")


if __name__ == "__main__":
    main()
