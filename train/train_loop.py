"""Minimal fine-tuning loop for the YOLOX experiment written by finetune_yolox.py.

YOLOX's own trainer ran at 5 s/iteration on the Windows dev laptop (GPU idle, workers idle;
the loader and the GPU step measured 0.06 s and 0.13 s), so this loop does the same job
without it: exp data loader (mosaic + HSV + flip), exp model initialised from the COCO
checkpoint, SGD + warmup + cosine, fp32, EMA, COCO AP evaluation on the val split every few
epochs, best checkpoint saved in YOLOX's own format so `finetune_yolox.py export` works.

    C:\\dev\\venv-train\\Scripts\\python train\\train_loop.py --exp train\\runs\\exp_yolox_tiny.py
        --ckpt train\\runs\\yolox_tiny_coco.pth --out C:\\dev\\vr-train\\runs [--epochs 15] [--batch 8] [--lr 0.0025]
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.getcwd())


def load_exp(path: str):
    spec = importlib.util.spec_from_file_location("exp_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Exp()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--ckpt", required=True, help="COCO-pretrained yolox checkpoint to start from")
    ap.add_argument("--out", default="C:/dev/vr-train/runs")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.0025, help="peak lr for batch 8 (YOLOX default 0.01/64 per image)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--no-aug-last", type=int, default=2, help="last N epochs without mosaic")
    a = ap.parse_args()

    from yolox.utils import ModelEMA
    exp = load_exp(a.exp)
    exp.data_num_workers = a.workers
    out = Path(a.out) / exp.exp_name
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    model = exp.get_model()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    state = ck.get("model", ck)
    own = model.state_dict()
    loaded = 0
    for k, v in state.items():
        if k in own and own[k].shape == v.shape:
            own[k] = v
            loaded += 1
    model.load_state_dict(own)
    print(f"loaded {loaded}/{len(own)} tensors from {a.ckpt} (head class layers re-initialised)")
    model.to(dev).train()
    ema = ModelEMA(model, 0.9998)

    loader = exp.get_data_loader(batch_size=a.batch, is_distributed=False, no_aug=False, cache_img=None)
    iters = len(loader)
    total = iters * a.epochs
    warm = iters  # one epoch warm-up
    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, torch.nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, torch.nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, torch.nn.Parameter):
            pg1.append(v.weight)
    opt = torch.optim.SGD(pg0, lr=a.lr, momentum=0.9, nesterov=True)
    opt.add_param_group({"params": pg1, "weight_decay": 5e-4})
    opt.add_param_group({"params": pg2})

    def lr_at(step: int) -> float:
        if step < warm:
            return a.lr * (step / warm) ** 2
        p = (step - warm) / max(1, total - warm)
        return 0.05 * a.lr + 0.95 * a.lr * 0.5 * (1 + math.cos(math.pi * p))

    evaluator = exp.get_evaluator(batch_size=a.batch, is_distributed=False)
    best = -1.0
    step = 0
    t_start = time.time()
    for epoch in range(a.epochs):
        if epoch == a.epochs - a.no_aug_last:
            loader.close_mosaic()
            model.head.use_l1 = True
            print("mosaic off for the final epochs")
        t_ep = time.time()
        run_loss = 0.0
        for i, batch in enumerate(loader):
            imgs, targets = batch[0], batch[1]
            imgs = imgs.to(dev, non_blocking=True).float()
            targets = targets.to(dev, non_blocking=True).float()
            lr = lr_at(step)
            for g in opt.param_groups:
                g["lr"] = lr
            outputs = model(imgs, targets)
            loss = outputs["total_loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ema.update(model)
            run_loss += float(loss)
            step += 1
            if (i + 1) % 100 == 0:
                el = time.time() - t_ep
                print(f"epoch {epoch + 1}/{a.epochs} iter {i + 1}/{iters} loss {run_loss / (i + 1):.2f} lr {lr:.5f} "
                      f"{el / (i + 1):.2f}s/it", flush=True)
        print(f"epoch {epoch + 1} done in {(time.time() - t_ep) / 60:.1f} min, mean loss {run_loss / max(1, iters):.3f}", flush=True)
        torch.save({"model": ema.ema.state_dict(), "epoch": epoch}, out / "latest_ckpt.pth")
        if (epoch + 1) % a.eval_every == 0 or epoch + 1 == a.epochs:
            ema.ema.eval()
            ap50_95, ap50, summary = evaluator.evaluate(ema.ema, False, False)
            ema.ema.train()
            print(f"val after epoch {epoch + 1}: AP50:95 {ap50_95:.3f} AP50 {ap50:.3f}", flush=True)
            print(summary[:600])
            if ap50 > best:
                best = ap50
                torch.save({"model": ema.ema.state_dict(), "epoch": epoch, "ap50": ap50}, out / "best_ckpt.pth")
                print("saved best", flush=True)
    print(f"finished in {(time.time() - t_start) / 60:.1f} min, best AP50 {best:.3f}; checkpoints in {out}")


if __name__ == "__main__":
    main()
