"""Open-vocabulary detector with text prompts (evaluation adapter).

Backend: Grounding DINO tiny (IDEA-Research, Apache-2.0) through Hugging Face
transformers, CPU. Prompts are plain phrases ("laptop screen", "license
plate"); each phrase becomes a label. It is slow on CPU (hundreds of ms to a
few seconds per frame) so it is meant for a low detection rate (every 1-2 s)
combined with carry-tracking, or for the offline high-accuracy mode. Requires
the evaluation venv (torch + transformers); the model (~700 MB) is downloaded
on first use into <models>/eval/hf.

Alternatives considered: YOLOE (Ultralytics, AGPL - licence conflict),
YOLO-World (GPL code), OWLv2 (Apache, slower than GDINO-tiny on CPU),
RF-DETR (Apache but closed-vocabulary COCO classes).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from videoredact.core.model import BBox
from videoredact.paths import models_dir
from .base import Detection, Detector

DEFAULT_PROMPTS = ["computer monitor", "laptop screen", "television", "phone screen", "tablet", "notebook",
                   "paper document", "clipboard", "whiteboard", "ID card", "credit card", "license plate"]
# prompt -> VideoRedact label
PROMPT_LABELS = {"computer monitor": "screen", "laptop screen": "screen", "television": "screen", "tablet": "screen",
                 "phone screen": "phone", "notebook": "document", "paper document": "document", "clipboard": "document",
                 "whiteboard": "document", "ID card": "id_card", "credit card": "id_card", "license plate": "plate"}


class GroundingDinoDetector(Detector):
    name = "gdino-tiny"

    def __init__(self, prompts: Optional[Sequence[str]] = None, score_thr: float = 0.3, text_thr: float = 0.25,
                 model_id: str = "IDEA-Research/grounding-dino-tiny", max_side: int = 800):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        cache = models_dir() / "eval" / "hf"
        cache.mkdir(parents=True, exist_ok=True)
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=str(cache))
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, cache_dir=str(cache)).eval()
        self.prompts = list(prompts or DEFAULT_PROMPTS)
        self.text = ". ".join(p.lower() for p in self.prompts) + "."
        self.score_thr, self.text_thr, self.max_side = score_thr, text_thr, max_side
        self.labels = {PROMPT_LABELS.get(p, p.replace(" ", "_")) for p in self.prompts}

    def detect(self, frames: Sequence[np.ndarray]) -> list[list[Detection]]:
        import cv2
        from PIL import Image
        out = []
        for f in frames:
            H, W = f.shape[:2]
            s = min(1.0, self.max_side / max(H, W))
            img = cv2.resize(f, (int(W * s), int(H * s))) if s < 1 else f
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            inputs = self.processor(images=pil, text=self.text, return_tensors="pt")
            with self.torch.no_grad():
                res = self.model(**inputs)
            post = self.processor.post_process_grounded_object_detection(
                res, inputs.input_ids, threshold=self.score_thr, text_threshold=self.text_thr,
                target_sizes=[(img.shape[0], img.shape[1])])[0]
            dets = []
            labels = post.get("text_labels") or post.get("labels")
            for box, score, label in zip(post["boxes"], post["scores"], labels):
                x1, y1, x2, y2 = (box / s).tolist()
                lab = str(label).strip()
                mapped = next((PROMPT_LABELS.get(p, p.replace(" ", "_")) for p in self.prompts if p.lower() in lab or lab in p.lower()), lab)
                dets.append(Detection(BBox.from_pixels(x1, y1, x2, y2, W, H), mapped, float(score), self.name))
            out.append(dets)
        return out
