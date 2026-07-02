"""
Единый реестр моделей: разводит две «семьи» детекторов.

  - torchvision : faster_rcnn, ssd, retinanet  (обучаются нашим train.py)
  - ultralytics : yolov8, yolov5             (обучаются своим .train)
"""
from __future__ import annotations

from src.models.torchvision_models import TORCHVISION_BUILDERS
from src.models.ultralytics_models import ULTRALYTICS_WEIGHTS

TORCHVISION_MODELS = set(TORCHVISION_BUILDERS)
ULTRALYTICS_MODELS = set(ULTRALYTICS_WEIGHTS)
ALL_MODELS = sorted(TORCHVISION_MODELS | ULTRALYTICS_MODELS)


def model_family(name: str) -> str:
    if name in TORCHVISION_MODELS:
        return "torchvision"
    if name in ULTRALYTICS_MODELS:
        return "ultralytics"
    raise KeyError(f"Неизвестная модель '{name}'. Доступны: {ALL_MODELS}")


def build_torchvision_model(name: str, num_classes: int):
    """num_classes ВКЛЮЧАЕТ фон (= len(classes) + 1)."""
    return TORCHVISION_BUILDERS[name](num_classes)
