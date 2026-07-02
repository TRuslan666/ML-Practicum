"""
Единая оценка и сравнение всех обученных моделей на валидации.
Результат:
  - results/logs/summary.csv         — таблица метрик по моделям;
  - results/plots/model_comparison.png, confusion_<model>.png, training_curves.png
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.dataset.dataset import load_classes
from src.evaluation.metrics import compute_map, compute_prf1, confusion_matrix
from src.models.registry import model_family
from src.utils.plots import plot_comparison_bar, plot_confusion, plot_training_curves
from src.utils.utils import ensure_dir, get_device, get_logger

log = get_logger("compare")
_EVAL_CONF = 0.001  # низкий порог для сбора предсказаний


def _load_val(cfg: dict):
    """Возвращает список записей валидации (путь, gt-боксы, gt-метки 0-индекс)."""
    path = Path(cfg["data"]["processed"]) / "gtsdb" / "manifest_val.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _targets_from_manifest(records) -> list[dict]:
    targets = []
    for r in records:
        targets.append(
            {
                "boxes": torch.tensor(r["boxes"], dtype=torch.float32),
                "labels": torch.tensor(r["labels"], dtype=torch.int64),  # уже 0-индекс
            }
        )
    return targets


def _predict_torchvision(model_name: str, cfg: dict, records) -> list[dict]:
    from src.utils.inference import TorchvisionDetector

    ckpt = Path(cfg["output"]["checkpoints"]) / model_name / "best.pt"
    det = TorchvisionDetector(model_name, str(ckpt), cfg)
    preds = []
    for r in tqdm(records, desc=f"{model_name} eval", leave=False):
        img = cv2.imread(r["image"])
        dets = det.predict(img, conf=_EVAL_CONF)
        if dets:
            boxes = torch.tensor([d[0] for d in dets], dtype=torch.float32)
            scores = torch.tensor([d[1] for d in dets], dtype=torch.float32)
            labels = torch.tensor([d[2] for d in dets], dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4)); scores = torch.zeros((0,)); labels = torch.zeros((0,), dtype=torch.int64)
        preds.append({"boxes": boxes, "scores": scores, "labels": labels})
    return preds


def _predict_ultralytics(model_name: str, cfg: dict, records) -> list[dict]:
    from src.utils.inference import _load_ultra

    # Канонический чекпойнт; откат на родную папку ultralytics, если копии нет.
    ckpt = Path(cfg["output"]["checkpoints"]) / model_name / "best.pt"
    if not ckpt.exists():
        ckpt = Path(cfg["output"]["logs"]) / "ultralytics" / model_name / "weights" / "best.pt"
    ckpt = ckpt.resolve()
    model = _load_ultra(model_name, str(ckpt))
    device = 0 if get_device(cfg["train"]["device"]).type == "cuda" else "cpu"
    preds = []
    for r in tqdm(records, desc=f"{model_name} eval", leave=False):
        res = model.predict(r["image"], conf=_EVAL_CONF, device=device, verbose=False)[0]
        b = res.boxes
        if b is not None and len(b):
            preds.append(
                {
                    "boxes": b.xyxy.cpu(),
                    "scores": b.conf.cpu(),
                    "labels": b.cls.cpu().to(torch.int64),  # ultralytics уже 0-индекс
                }
            )
        else:
            preds.append({"boxes": torch.zeros((0, 4)), "scores": torch.zeros((0,)), "labels": torch.zeros((0,), dtype=torch.int64)})
    return preds


def evaluate_models(model_names: list[str], cfg: dict) -> pd.DataFrame:
    classes = load_classes(cfg)
    records = _load_val(cfg)
    targets = _targets_from_manifest(records)
    iou_thr = cfg["eval"]["iou_threshold"]
    conf_thr = cfg["eval"]["conf_threshold"]

    rows = []
    for name in model_names:
        log.info("Оцениваю %s ...", name)
        if model_family(name) == "ultralytics":
            preds = _predict_ultralytics(name, cfg, records)
        else:
            preds = _predict_torchvision(name, cfg, records)

        m = compute_map(preds, targets)
        prf = compute_prf1(preds, targets, iou_thr, conf_thr)
        rows.append({"model": name, **m, **prf})

        # матрица ошибок
        cm = confusion_matrix(preds, targets, len(classes), iou_thr, conf_thr)
        plot_confusion(cm, classes, name, cfg["output"]["plots"])
        log.info("  %s: mAP@0.5=%.3f F1=%.3f", name, m["mAP@0.5"], prf["f1"])

    df = pd.DataFrame(rows).sort_values("mAP@0.5", ascending=False)
    summary_path = Path(cfg["output"]["logs"]) / "summary.csv"
    ensure_dir(Path(cfg["output"]["logs"]))
    df.to_csv(summary_path, index=False)
    log.info("Сводка метрик -> %s", summary_path)

    # графики сравнения и кривых обучения
    plot_comparison_bar(str(summary_path), cfg["output"]["plots"])
    plot_training_curves(model_names, cfg["output"]["logs"], cfg["output"]["plots"])

    print("\n=== ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ ===")
    print(df.to_string(index=False))
    return df
