"""Метрики качества детекции: mAP, Precision, Recall, F1 + матрица ошибок.

Единый формат предсказаний и таргетов (как в torchvision detection):
    pred   = {"boxes": Tensor[N,4] xyxy, "scores": Tensor[N], "labels": Tensor[N]}
    target = {"boxes": Tensor[M,4] xyxy, "labels": Tensor[M]}
Метки классов здесь — 0-индексированные (без фона); конвертацию из моделей
делаем на стороне инференса/обучения.
"""
from __future__ import annotations

import numpy as np
import torch
from torchvision.ops import box_iou


def compute_map(preds: list[dict], targets: list[dict]) -> dict[str, float]:
    """mAP@0.5, mAP@0.5:0.95 через torchmetrics."""
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    metric.update(preds, targets)
    res = metric.compute()
    return {
        "mAP@0.5": float(res["map_50"]),
        "mAP@0.5:0.95": float(res["map"]),
        "mAR@100": float(res["mar_100"]),
    }


def _match_image(pred: dict, target: dict, iou_thr: float, conf_thr: float):
    """Жадное сопоставление предсказаний и GT на одном изображении.

    Возвращает списки (tp, fp) по предсказаниям и число непокрытых GT (fn),
    а также пары (gt_label, pred_label) для матрицы ошибок.
    """
    keep = pred["scores"] >= conf_thr
    pboxes = pred["boxes"][keep]
    plabels = pred["labels"][keep]
    pscores = pred["scores"][keep]

    gboxes = target["boxes"]
    glabels = target["labels"]

    order = torch.argsort(pscores, descending=True)
    pboxes, plabels = pboxes[order], plabels[order]

    matched_gt = set()
    tp = fp = 0
    pairs = []  # (gt_label, pred_label) для верно локализованных по IoU
    if len(gboxes) == 0:
        return 0, len(pboxes), 0, pairs

    ious = box_iou(pboxes, gboxes) if len(pboxes) else torch.zeros((0, len(gboxes)))
    for i in range(len(pboxes)):
        best_iou, best_j = (ious[i].max(0) if len(gboxes) else (torch.tensor(0.0), None))
        if best_j is not None and float(best_iou) >= iou_thr and int(best_j) not in matched_gt:
            matched_gt.add(int(best_j))
            pairs.append((int(glabels[best_j]), int(plabels[i])))
            if int(plabels[i]) == int(glabels[best_j]):
                tp += 1
            else:
                fp += 1  # локализован, но класс неверный
        else:
            fp += 1
    fn = len(gboxes) - len(matched_gt)
    return tp, fp, fn, pairs


def compute_prf1(
    preds: list[dict],
    targets: list[dict],
    iou_thr: float = 0.5,
    conf_thr: float = 0.25,
) -> dict[str, float]:
    """Микро-усреднённые Precision / Recall / F1 по всему набору."""
    TP = FP = FN = 0
    for p, t in zip(preds, targets):
        tp, fp, fn, _ = _match_image(p, t, iou_thr, conf_thr)
        TP += tp
        FP += fp
        FN += fn
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def confusion_matrix(
    preds: list[dict],
    targets: list[dict],
    num_classes: int,
    iou_thr: float = 0.5,
    conf_thr: float = 0.25,
) -> np.ndarray:
    """Матрица ошибок [num_classes, num_classes] (строки — GT, столбцы — предсказание).

    Считаются только корректно локализованные (IoU >= порог) объекты —
    показывает, какие классы модель путает между собой.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, t in zip(preds, targets):
        _, _, _, pairs = _match_image(p, t, iou_thr, conf_thr)
        for gt_label, pred_label in pairs:
            if 0 <= gt_label < num_classes and 0 <= pred_label < num_classes:
                cm[gt_label, pred_label] += 1
    return cm


def evaluate_all(
    preds: list[dict],
    targets: list[dict],
    iou_thr: float = 0.5,
    conf_thr: float = 0.25,
) -> dict[str, float]:
    """Полный набор метрик одной моделью одним вызовом."""
    out = compute_map(preds, targets)
    out.update(compute_prf1(preds, targets, iou_thr, conf_thr))
    return out
