"""
Обёртка над моделями ultralytics: YOLOv8 и YOLOv5

У этих моделей собственный высокоуровневый цикл обучения (.train) и
валидации (.val), поэтому они обучаются отдельно от torchvision-детекторов,
но на ТОМ ЖЕ датасете (data/processed/gtsdb/yolo/data.yaml) и с теми же классами.
"""
from __future__ import annotations

ULTRALYTICS_WEIGHTS = {
    "yolov8": "yolov8s.pt",   
    "yolov5": "yolov5su.pt"
}


def build_ultralytics(name: str):
    """Создаёт модель ultralytics с предобученными весами."""
    if name not in ULTRALYTICS_WEIGHTS:
        raise KeyError(f"Неизвестная ultralytics-модель: {name}")
    weights = ULTRALYTICS_WEIGHTS[name]
    from ultralytics import YOLO
    return YOLO(weights)
