"""
    Циклы обучения для обеих семей моделей.

  - train_torchvision : ручной цикл для Faster R-CNN / SSD / RetinaNet
  - train_ultralytics : обёртка над .train() для YOLO

"""
from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataset.dataset import build_dataloaders, load_classes
from src.evaluation.metrics import compute_map, compute_prf1
from src.models.registry import build_torchvision_model
from src.models.ultralytics_models import build_ultralytics
from src.utils.utils import ensure_dir, get_device, get_logger


# --------------------------------------------------------------------------- #
#  torchvision-детекторы (ручной цикл)
# --------------------------------------------------------------------------- #
def _build_optimizer(model, tcfg):
    params = [p for p in model.parameters() if p.requires_grad]
    if tcfg["optimizer"].lower() == "sgd":
        return torch.optim.SGD(
            params, lr=tcfg["lr"], momentum=tcfg["momentum"], weight_decay=tcfg["weight_decay"]
        )
    return torch.optim.AdamW(params, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])


def _build_scheduler(optimizer, tcfg, steps_per_epoch):
    kind = tcfg.get("lr_scheduler", "none")
    epochs = tcfg["epochs"]
    warmup = int(tcfg.get("warmup_epochs", 0) or 0)
    if kind == "cosine":
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - warmup)
        )
        if warmup > 0:
            # Линейный прогрев LR с почти нуля
            warm = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warm, cosine], milestones=[warmup]
            )
        return cosine
    if kind == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    return None


@torch.no_grad()
def _collect_predictions(model, loader, device):
    """Прогон по валидации -> (preds, targets) в 0-индексированном формате метрик."""
    model.eval()
    preds, targets = [], []
    for images, batch_targets in tqdm(loader, desc="val", leave=False):
        images = [img.to(device) for img in images]
        outputs = model(images)
        for out, tgt in zip(outputs, batch_targets):
            preds.append(
                {
                    "boxes": out["boxes"].cpu(),
                    "scores": out["scores"].cpu(),
                    "labels": (out["labels"].cpu() - 1), 
                }
            )
            targets.append(
                {
                    "boxes": tgt["boxes"],
                    "labels": (tgt["labels"] - 1),
                }
            )
    return preds, targets


def train_torchvision(model_name: str, cfg: dict) -> dict:
    tcfg = cfg["train"]
    device = get_device(tcfg["device"])
    out_dir = ensure_dir(Path(cfg["output"]["checkpoints"]) / model_name)
    log = get_logger(model_name, Path(cfg["output"]["logs"]) / f"{model_name}.log")
    writer = SummaryWriter(Path(cfg["output"]["logs"]) / "torchvision" / model_name)

    classes = load_classes(cfg)
    num_classes = len(classes) + 1  
    model = build_torchvision_model(model_name, num_classes).to(device)

    train_loader, val_loader = build_dataloaders(cfg)
    optimizer = _build_optimizer(model, tcfg)
    scheduler = _build_scheduler(optimizer, tcfg, len(train_loader))

    # AMP экономит VRAM, но focal loss RetinaNet под fp16 численно нестабилен
    # (log/exp переполняются -> inf/NaN, обучение недетерминированно срывается).
    # Для RetinaNet считаем в fp32
    use_amp = bool(tcfg["amp"]) and device.type == "cuda" and model_name != "retinanet"
    if tcfg["amp"] and model_name == "retinanet":
        log.info("AMP отключён для retinanet")
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = []
    best_map = -1.0
    patience = tcfg.get("early_stopping_patience", 10)
    bad_epochs = 0

    log.info("=== Обучение %s | классов = %d | устройство = %s ===", model_name, len(classes), device)
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        running = 0.0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"{model_name} e{epoch}/{tcfg['epochs']}")
        for images, batch_targets in pbar:
            images = [img.to(device) for img in images]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in batch_targets]

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss_dict = model(images, tgts)
                loss = sum(loss_dict.values())

            if not math.isfinite(loss.item()):
                log.warning("Проблемный loss, пропуск шага")
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            # Клиппинг по норме (на размасштабированных градиентах) гасит редкие
            # всплески и не даёт обучению уйти в NaN. unscale_/clip — no-op при AMP off.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        if scheduler:
            scheduler.step()
        train_loss = running / max(1, len(train_loader))

        #Валидация
        preds, targets = _collect_predictions(model, val_loader, device)
        m = compute_map(preds, targets)
        prf = compute_prf1(preds, targets, cfg["eval"]["iou_threshold"], cfg["eval"]["conf_threshold"])
        epoch_rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "time_s": round(time.time() - t0, 1),
            **m,
            **prf,
        }
        history.append(epoch_rec)
        for k, v in epoch_rec.items():
            if k != "epoch":
                writer.add_scalar(k, v, epoch)
        log.info(
            "e%02d | loss=%.3f | mAP@0.5=%.3f | mAP=%.3f | F1=%.3f | %.0fs",
            epoch, train_loss, m["mAP@0.5"], m["mAP@0.5:0.95"], prf["f1"], epoch_rec["time_s"],
        )

        #Чекпойнт лучшего + early stopping по mAP@0.5
        if m["mAP@0.5"] > best_map:
            best_map = m["mAP@0.5"]
            torch.save(model.state_dict(), out_dir / "best.pt")
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log.info("Early stopping на эпохе %d (нет улучшений %d эпох)", epoch, patience)
                break

    # сохраняем историю
    hist_path = Path(cfg["output"]["logs"]) / f"{model_name}_history.json"
    hist_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    writer.close()
    log.info("Готово %s | лучший mAP@0.5=%.3f | история -> %s", model_name, best_map, hist_path)
    return {"model": model_name, "best_mAP@0.5": best_map, "history": str(hist_path)}


# --------------------------------------------------------------------------- #
#  ultralytics (YOLOv8 / YOLOv5)
# --------------------------------------------------------------------------- #
def train_ultralytics(model_name: str, cfg: dict) -> dict:
    tcfg = cfg["train"]
    device = 0 if get_device(tcfg["device"]).type == "cuda" else "cpu"
    data_yaml = Path(cfg["data"]["processed"]) / "gtsdb" / "yolo" / "data.yaml"
    log = get_logger(model_name, Path(cfg["output"]["logs"]) / f"{model_name}.log")

    model = build_ultralytics(model_name)
    log.info("=== Обучение %s (ultralytics) | устройство=%s ===", model_name, device)

    results = model.train(
        data=str(data_yaml),
        epochs=tcfg["epochs"],
        imgsz=cfg["data"]["img_size"],
        batch=tcfg["batch_size"],
        optimizer="AdamW" if tcfg["optimizer"].lower() == "adamw" else "SGD",
        lr0=tcfg["lr"],
        weight_decay=tcfg["weight_decay"],
        device=device,
        amp=tcfg["amp"],
        patience=tcfg.get("early_stopping_patience", 10),
        seed=cfg.get("seed", 42),
        project=str((Path(cfg["output"]["logs"]) / "ultralytics").resolve()),
        name=model_name,
        exist_ok=True,
        verbose=True,
    )
    # ultralytics сам пишет метрики/графики в project/name/.
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else None

    # Дублируем лучший вес в единое место results/checkpoints/<model>/best.pt,
    # чтобы веса всех моделей лежали в одном предсказуемом месте (оригинал
    # ultralytics со своими графиками остаётся нетронутым).
    src_best = Path(cfg["output"]["logs"]) / "ultralytics" / model_name / "weights" / "best.pt"
    if src_best.exists():
        dst = ensure_dir(Path(cfg["output"]["checkpoints"]) / model_name) / "best.pt"
        shutil.copy2(src_best, dst)
        log.info("Веса продублированы -> %s", dst)

    log.info("Готово %s | результаты ultralytics -> %s", model_name, save_dir)
    return {"model": model_name, "save_dir": str(save_dir)}


def train_model(model_name: str, cfg: dict) -> dict:
    """Диспетчер: выбирает нужный цикл обучения по семье модели."""
    from src.models.registry import model_family

    if model_family(model_name) == "ultralytics":
        return train_ultralytics(model_name, cfg)
    return train_torchvision(model_name, cfg)
