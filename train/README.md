# Fine-tuning the object detector on our footage

Why: COCO classes do not know "ID card", "license plate", "notepad" or "back of a
head". A detector trained on a few hundred of our own labelled frames (plus
permissive public data) replaces the COCO mapping and the head heuristic with
real classes, at no extra runtime cost.

What you need
- Labels from the benchmark workflow (`bench/README.md`): `bench/labels/<clip>/*.txt` (YOLO) and
  `bench/frames/<clip>/*.jpg`. Aim for 300+ boxes per class; add a `head` class when labeling
  (draw the whole head, including turned-away heads) so the fallback becomes learned.
- A GPU (the dev laptop's RTX 3050 Ti is enough for yolox_tiny at batch 8-16) or patience.
- The training venv: `pip install yolox torch torchvision` (Apache-2.0; use the PyTorch CUDA
  wheel matching the driver). Do **not** use Ultralytics (AGPL).

Steps
```powershell
C:\dev\venv-eval\Scripts\python train\finetune_yolox.py prepare           # YOLO labels -> COCO + split by clip
C:\dev\venv-eval\Scripts\python train\finetune_yolox.py train --model yolox_tiny --epochs 60 --batch 16
C:\dev\venv-eval\Scripts\python train\finetune_yolox.py export --ckpt train\runs\yolox_tiny_videoredact\best_ckpt.pth --out models\yolox_tiny_vr.onnx
```
Public data to add (all CC BY 4.0 / CC0; download manually into `train/extra/<name>/` in YOLO
layout and point `prepare --labels` at a merged folder): Roboflow Universe head-detection sets,
Roboflow "License Plates" (Open Images subset), the CC0 Kaggle plate set. Record the source and
licence of every dataset in `train/DATA_SOURCES.md` before training.

Plugging the model in
1. Add a `ModelSpec` for `yolox_tiny_vr.onnx` in `videoredact/vision/models.py` (or place the file in
   `%LOCALAPPDATA%\VideoRedact\models` and the bundled `models\` folder).
2. The exported model's class order is in `models\yolox_tiny_vr.classes.json`; set
   `ObjectDetector(..., class_names=[...])` so ids map to labels (the detector falls back to the COCO
   mapping when no class list is given).
3. Re-run `bench/run_bench.py` and `bench/eval_faces.py`; commit the JSON next to the baseline.

Evaluation gate: ship only if `leak_rate` and per-class `recall@90` on held-out clips are
equal or better than the current pipeline, and precision does not fall by more than 10 points.
