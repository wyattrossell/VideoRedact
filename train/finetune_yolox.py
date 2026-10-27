"""Fine-tune YOLOX (Apache-2.0) on our labelled frames so the object detector learns
our classes directly: head, face, screen, phone_screen, document, plate, id_card.

Pipeline
  1. bench/labels (YOLO txt per frame) + bench/frames (jpg)  ->  COCO json + train/val split
  2. YOLOX training with the official trainer (pip install yolox), starting from the
     COCO-pretrained yolox_tiny/yolox_s weights (bundled download links below)
  3. export best checkpoint to ONNX in the input layout our ObjectDetector expects
     (1x3xHxW, 85+ outputs decoded by videoredact/vision/detector.py; class list in
     train/classes.json)

Usage (GPU recommended; CPU works for yolox_tiny but takes hours):
  python train/finetune_yolox.py prepare --frames bench/frames --labels bench/labels --out train/data
  python train/finetune_yolox.py train --data train/data --model yolox_tiny --epochs 60 --batch 16
  python train/finetune_yolox.py export --ckpt train/runs/yolox_tiny/best_ckpt.pth --model yolox_tiny --out models/yolox_tiny_vr.onnx

Then register the model in videoredact/vision/models.py and map its class ids in
COCO_TO_LABEL (or set `labels_file` in the ObjectDetector) - see train/README.md.
Licensing: YOLOX code and COCO weights Apache-2.0; our frames are agency footage
(never leave the machine); any extra public data must be CC BY / CC0.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

CLASSES = ["face", "screen", "phone_screen", "document", "plate", "id_card", "head"]
PRETRAINED = {
    "yolox_tiny": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth",
    "yolox_s": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth",
}


def prepare(frames: Path, labels: Path, out: Path, val_frac: float = 0.15, seed: int = 7) -> None:
    import cv2
    classes = CLASSES
    cf = labels / "classes.txt"
    if cf.exists():
        classes = [l.strip() for l in cf.read_text().splitlines() if l.strip()]
    items = []
    for txt in sorted(labels.rglob("*.txt")):
        if txt.name == "classes.txt":
            continue
        img = frames / txt.parent.name / (txt.stem + ".jpg")
        if img.exists():
            items.append((img, txt))
    if not items:
        sys.exit("no labelled frames found (run the labeling workflow in bench/README.md first)")
    random.Random(seed).shuffle(items)
    # split by clip so val clips are unseen
    clips = sorted({i[0].parent.name for i in items})
    random.Random(seed).shuffle(clips)
    val_clips = set(clips[: max(1, int(len(clips) * val_frac))])
    # YOLOX 0.3 hard-codes the image folder names train2017 / val2017 in its COCO loader
    for split in ("train2017", "val2017"):
        (out / split).mkdir(parents=True, exist_ok=True)
    coco = {s: {"images": [], "annotations": [], "categories": [{"id": i + 1, "name": c} for i, c in enumerate(classes)]} for s in ("train", "val")}
    ann_id = 1
    for img_id, (img, txt) in enumerate(items, 1):
        split = "val" if img.parent.name in val_clips else "train"
        dst = out / f"{split}2017" / f"{img.parent.name}__{img.name}"
        if not dst.exists():
            shutil.copy2(img, dst)
        im = cv2.imread(str(dst))
        H, W = im.shape[:2]
        coco[split]["images"].append({"id": img_id, "file_name": dst.name, "width": W, "height": H})
        for line in txt.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            c = int(float(p[0]))
            cx, cy, w, h = (float(v) for v in p[1:5])
            coco[split]["annotations"].append({"id": ann_id, "image_id": img_id, "category_id": c + 1,
                                               "bbox": [(cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H],
                                               "area": w * W * h * H, "iscrowd": 0})
            ann_id += 1
    (out / "annotations").mkdir(exist_ok=True)
    for s in ("train", "val"):
        (out / "annotations" / f"instances_{s}.json").write_text(json.dumps(coco[s]), encoding="utf-8")
    (out / "classes.json").write_text(json.dumps(classes), encoding="utf-8")
    print(f"train: {len(coco['train']['images'])} images / {len(coco['train']['annotations'])} boxes; "
          f"val: {len(coco['val']['images'])} / {len(coco['val']['annotations'])}; classes={classes}")


def write_exp(data: Path, model: str, epochs: int, batch: int, out_dir: Path) -> Path:
    classes = json.loads((data / "classes.json").read_text())
    depth, width = {"yolox_tiny": (0.33, 0.375), "yolox_s": (0.33, 0.50)}[model]
    size = 416 if model == "yolox_tiny" else 640
    exp = f'''# auto-generated YOLOX experiment
import os
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.exp_name = "{model}_videoredact"
        self.num_classes = {len(classes)}
        self.depth, self.width = {depth}, {width}
        self.input_size = ({size}, {size})
        self.test_size = ({size}, {size})
        self.data_dir = r"{data.resolve()}"
        self.train_ann = "instances_train.json"
        self.val_ann = "instances_val.json"
        self.train_name = "train2017"
        self.val_name = "val2017"
        self.max_epoch = {epochs}
        self.no_aug_epochs = max(2, {epochs} // 10)
        self.warmup_epochs = 1
        self.eval_interval = 2
        self.enable_mixup = False
        self.mosaic_scale = (0.5, 1.5)
        self.output_dir = r"{out_dir.resolve()}"
        self.data_num_workers = 4
'''
    p = out_dir / f"exp_{model}.py"
    out_dir.mkdir(parents=True, exist_ok=True)
    p.write_text(exp, encoding="utf-8")
    return p


def train(data: Path, model: str, epochs: int, batch: int, device_count: int) -> None:
    runs = Path("train/runs")
    exp = write_exp(data, model, epochs, batch, runs)
    ckpt = runs / f"{model}_coco.pth"
    if not ckpt.exists():
        import urllib.request
        print("downloading pretrained", PRETRAINED[model])
        urllib.request.urlretrieve(PRETRAINED[model], ckpt)
    cmd = [sys.executable, "-m", "yolox.tools.train", "-f", str(exp), "-d", str(device_count), "-b", str(batch),
           "--fp16", "-o", "-c", str(ckpt)]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def export(ckpt: Path, model: str, out: Path) -> None:
    data = Path("train/data")
    exp = write_exp(data, model, 1, 1, Path("train/runs"))
    cmd = [sys.executable, "-m", "yolox.tools.export_onnx", "-f", str(exp), "-c", str(ckpt), "--output-name", str(out),
           "--opset", "13"]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    shutil.copy2(data / "classes.json", out.with_suffix(".classes.json"))
    print("exported", out, "class list", out.with_suffix(".classes.json"))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--frames", default="bench/frames"); p.add_argument("--labels", default="bench/labels"); p.add_argument("--out", default="train/data")
    t = sub.add_parser("train"); t.add_argument("--data", default="train/data"); t.add_argument("--model", default="yolox_tiny", choices=list(PRETRAINED)); t.add_argument("--epochs", type=int, default=60); t.add_argument("--batch", type=int, default=16); t.add_argument("--devices", type=int, default=1)
    e = sub.add_parser("export"); e.add_argument("--ckpt", required=True); e.add_argument("--model", default="yolox_tiny"); e.add_argument("--out", default="models/yolox_tiny_vr.onnx")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(Path(a.frames), Path(a.labels), Path(a.out))
    elif a.cmd == "train":
        train(Path(a.data), a.model, a.epochs, a.batch, a.devices)
    else:
        export(Path(a.ckpt), a.model, Path(a.out))


if __name__ == "__main__":
    main()
