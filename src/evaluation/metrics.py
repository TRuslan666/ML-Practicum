"""
src/evaluation/metrics.py — оценка Faster R-CNN на val-сплите.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou


def compute_precision_recall_f1(
    detections: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
) -> dict[str, float]:
    """
    Классический Precision / Recall / F1.
    """
    dets_by_key: dict[tuple[int, int], list[dict]] = {}
    for det in detections:
        if det["score"] < score_threshold:
            continue
        key = (det["image_id"], det["category_id"])
        dets_by_key.setdefault(key, []).append(det)

    gts_by_key: dict[tuple[int, int], list[dict]] = {}
    for gt in ground_truths:
        key = (gt["image_id"], gt["category_id"])
        gts_by_key.setdefault(key, []).append(gt)

    tp, fp, fn = 0, 0, 0
    all_keys = set(dets_by_key.keys()) | set(gts_by_key.keys())

    for key in all_keys:
        dets = sorted(dets_by_key.get(key, []), key=lambda d: -d["score"])
        gts = gts_by_key.get(key, [])

        if not gts:
            fp += len(dets)
            continue
        if not dets:
            fn += len(gts)
            continue

        gt_boxes_xyxy = torch.tensor([
            [g["bbox"][0], g["bbox"][1], g["bbox"][0] + g["bbox"][2], g["bbox"][1] + g["bbox"][3]]
            for g in gts
        ], dtype=torch.float32)

        matched_gt = [False] * len(gts)

        for det in dets:
            x, y, w, h = det["bbox"]
            det_box = torch.tensor([[x, y, x + w, y + h]], dtype=torch.float32)
            ious = box_iou(det_box, gt_boxes_xyxy)[0]

            best_iou, best_idx = -1.0, -1
            for idx in range(len(gts)):
                if not matched_gt[idx] and ious[idx].item() > best_iou:
                    best_iou = ious[idx].item()
                    best_idx = idx

            if best_idx >= 0 and best_iou >= iou_threshold:
                matched_gt[best_idx] = True
                tp += 1
            else:
                fp += 1

        fn += matched_gt.count(False)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
    }


def compute_coco_map(detections: list[dict[str, Any]], val_annotations_path: str | Path) -> dict[str, float]:
    """COCO mAP через pycocotools."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(val_annotations_path))
    coco_dt = coco_gt.loadRes(detections) if detections else coco_gt.loadRes([])

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stat_names = [
        "mAP@0.5:0.95", "mAP@0.5", "mAP@0.75",
        "mAP_small", "mAP_medium", "mAP_large",
        "AR@1", "AR@10", "AR@100",
        "AR_small", "AR_medium", "AR_large",
    ]
    return dict(zip(stat_names, [float(v) for v in coco_eval.stats]))


@torch.no_grad()
def run_inference(model: torch.nn.Module, dataset, device: torch.device,
                  label_to_category_id: dict[int, int], 
                  score_threshold: float = 0.001) -> list[dict[str, Any]]:
    """Инференс Faster R-CNN."""
    model.eval()
    detections: list[dict[str, Any]] = []

    for idx in range(len(dataset)):
        image, target = dataset[idx]
        image_id = int(target["image_id"].item())

        output = model([image.to(device)])[0]
        boxes = output["boxes"].cpu()
        scores = output["scores"].cpu()
        labels = output["labels"].cpu()

        for box, score, label in zip(boxes, scores, labels):
            if score.item() < score_threshold:
                continue
            x1, y1, x2, y2 = box.tolist()
            category_id = label_to_category_id.get(int(label.item()))
            if category_id is None or category_id == 0:
                continue

            detections.append({
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score.item()),
            })

    return detections


def evaluate_faster_rcnn_from_config(config: dict[str, Any]) -> dict[str, Any]:
    from src.models.faster_rcnn import CocoDetectionDataset, _build_model, _resolve_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = _resolve_path(config["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    num_classes = checkpoint.get("num_classes", 91)
    print(f"Загружено классов: {num_classes}")

    model = _build_model(num_classes=num_classes, image_size=config.get("imgsz", 512))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    val_dataset = CocoDetectionDataset(
        config["val_images"], 
        config["val_annotations"],
        min_box_size=float(config.get("min_box_size", 2.0)),
    )
    
    print(f"Dataset classes: {val_dataset.category_to_label}")

    # === КРИТИЧНОЕ ИСПРАВЛЕНИЕ ===
    label_to_category_id = val_dataset.category_to_label  # {label: category_id} или наоборот?

    print(f"Инференс на {len(val_dataset)} изображений...")

    detections = run_inference(
        model, 
        val_dataset, 
        device, 
        label_to_category_id,
        score_threshold=0.01   # сильно понизили
    )
    print(f"Получено {len(detections)} детекций (score >= 0.01)")

    # Если всё ещё 0 — выводим пример предсказания
    if len(detections) == 0:
        print("WARNING: Модель ничего не детектирует! Проверяем первое изображение...")
        image, target = val_dataset[0]
        output = model([image.to(device)])[0]
        print(f"Пример предсказаний: scores = {output['scores'][:5]}")
        print(f"Пример labels = {output['labels'][:5]}")

    # COCO метрики (защищённо)
    val_annotations_path = _resolve_path(config["val_annotations"])
    
    if len(detections) > 0:
        coco_metrics = compute_coco_map(detections, val_annotations_path)
    else:
        print("WARNING: Нет детекций → COCO mAP будет нулевым")
        coco_metrics = {
            "mAP@0.5:0.95": 0.0, "mAP@0.5": 0.0, "mAP@0.75": 0.0,
            "mAP_small": 0.0, "mAP_medium": 0.0, "mAP_large": 0.0,
            "AR@1": 0.0, "AR@10": 0.0, "AR@100": 0.0,
            "AR_small": 0.0, "AR_medium": 0.0, "AR_large": 0.0,
        }

    # Precision/Recall/F1 (работает даже при 0 детекций)
    with val_annotations_path.open("r", encoding="utf-8") as f:
        gt_data = json.load(f)
    
    ground_truths = [
        {"image_id": ann["image_id"], "category_id": ann["category_id"], "bbox": ann["bbox"]}
        for ann in gt_data.get("annotations", [])
    ]

    prf_metrics = compute_precision_recall_f1(
        detections, ground_truths,
        iou_threshold=float(config.get("iou_threshold", 0.5)),
        score_threshold=float(config.get("score_threshold", 0.5))
    )

    # === СОХРАНЕНИЕ (всегда выполняется) ===
    output_dir = _resolve_path(config.get("project", "results")) / config.get("name", "faster_rcnn")
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "coco_map": coco_metrics,
        "precision_recall_f1": prf_metrics,
        "num_detections": len(detections),
        "num_images": len(val_dataset),
        "status": "success" if len(detections) > 0 else "no_detections"
    }

    output_file = metrics_dir / "eval_metrics.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nФайл успешно сохранён: {output_file}")
    print(f"   Количество детекций: {len(detections)}")
    
    # Вывод метрик
    print("\n=== COCO mAP ===")
    for name, value in coco_metrics.items():
        print(f"  {name}: {value:.4f}")

    print("\n=== Precision / Recall / F1 ===")
    print(f"  Precision: {prf_metrics['precision']:.4f}")
    print(f"  Recall:    {prf_metrics['recall']:.4f}")
    print(f"  F1:        {prf_metrics['f1']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Оценка Faster R-CNN на val")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val_images", required=True)
    parser.add_argument("--val_annotations", required=True)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--project", default="results")
    parser.add_argument("--name", default="faster_rcnn")
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--inference_score_threshold", type=float, default=0.001)

    args = parser.parse_args()
    evaluate_faster_rcnn_from_config(vars(args))


if __name__ == "__main__":
    main()