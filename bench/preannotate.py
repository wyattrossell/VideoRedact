"""Pre-annotate extracted frames with the current detectors to speed up manual labeling.

Writes a Label Studio import file (tasks with `predictions`) so the labeler only
corrects boxes instead of drawing every one. Predictions are NOT ground truth:
the benchmark must use the corrected, human-reviewed export.

Usage:  python bench/preannotate.py --frames bench/frames --out bench/labelstudio/tasks.json
Then in Label Studio: Import -> upload tasks.json (images are referenced by local path;
start Label Studio with LOCAL_FILES_SERVING_ENABLED=1, see bench/README.md).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from videoredact.vision.detector import CombinedDetector  # noqa: E402

LABEL_MAP = {"face": "face", "screen": "screen", "phone": "phone_screen", "document": "document"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="bench/frames")
    ap.add_argument("--out", default="bench/labelstudio/tasks.json")
    ap.add_argument("--root-url", default="/data/local-files/?d=", help="Label Studio local-files prefix")
    a = ap.parse_args()
    det = CombinedDetector(["face", "screen", "phone", "document"], face_conf=0.4, obj_conf=0.3, obj_model="yolox_s")
    frames_root = Path(a.frames).resolve()
    tasks = []
    for img in sorted(frames_root.rglob("*.jpg")):
        f = cv2.imread(str(img))
        if f is None:
            continue
        H, W = f.shape[:2]
        results = []
        for d in det.detect(f):
            results.append({
                "from_name": "box", "to_name": "image", "type": "rectanglelabels", "original_width": W, "original_height": H,
                "value": {"x": float(d.bbox.x) * 100, "y": float(d.bbox.y) * 100, "width": float(d.bbox.w) * 100, "height": float(d.bbox.h) * 100,
                          "rotation": 0, "rectanglelabels": [LABEL_MAP.get(d.label, d.label)]},
                "score": round(float(d.score), 3),
            })
        rel = img.relative_to(frames_root).as_posix()
        tasks.append({"data": {"image": f"{a.root_url}{rel}"},
                      "predictions": [{"model_version": "videoredact-current", "result": results}]})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(tasks, indent=1), encoding="utf-8")
    print(f"wrote {len(tasks)} tasks with predictions to {a.out}")


if __name__ == "__main__":
    main()
