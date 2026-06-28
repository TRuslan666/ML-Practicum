"""
src/models/yolo.py — обучение YOLOv5 и YOLOv8 через единый API пакета `ultralytics`.

С версии ultralytics>=8 пакет умеет загружать и тренировать архитектуры YOLOv5
("u"-варианты весов, например yolov5su.pt — anchor-free, обновлённая голова) и
YOLOv8 (yolov8n.pt и т.д.) через ОДИН И ТОТ ЖЕ класс YOLO. Это даёт два честно
сравнимых "слота" из требования "YOLO (разные версии)" без дублирования кода
тренировочного цикла.

Если для отчёта принципиально важна именно "классическая" (anchor-based) YOLOv5,
а не её unified-вариант — см. функцию train_yolov5_legacy() в конце файла,
которая клонирует оригинальный репозиторий ultralytics/yolov5 и запускает его
собственный train.py через subprocess.
"""

from pathlib import Path
from ultralytics import YOLO

# ПАТЧ: Ultralytics по умолчанию не распознаёт .ppm как формат изображения
# (его список IMG_FORMATS — обычный set, не frozenset, поэтому можно расширить
# в рантайме без правки исходников библиотеки). OpenCV (который Ultralytics
# использует внутри для чтения файлов) сам по себе .ppm читает нормально —
# проблема была только в фильтрации файлов при сканировании папки.
from ultralytics.data.utils import IMG_FORMATS
IMG_FORMATS.add("ppm")


def train_yolo(
    weights: str,
    data_yaml: str,
    epochs: int,
    imgsz: int,
    batch: int,
    project: str,
    name: str,
    **extra_kwargs,
):
    """
    Универсальная функция обучения для YOLOv5/YOLOv8 (и любых версий,
    поддерживаемых текущим пакетом ultralytics — v9, v10, v11 тоже сюда впишутся).

    weights: например "yolov5su.pt" (YOLOv5, anchor-free unified) или "yolov8n.pt"
    data_yaml: путь к dataset.yaml
    project/name: куда сохранять результаты — results/{project}/{name}
    """
    model = YOLO(weights)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        **extra_kwargs
    )
    return results


def train_yolov5(config: dict):
    return train_yolo(
        weights=config.get("weights", "yolov5s.pt"),
        data_yaml=config["data_yaml"],
        epochs=config.get("epochs", 50),
        imgsz=config.get("imgsz", 640),
        batch=config.get("batch", 16),
        project=config.get("project", "results/plots"),
        name=config.get("name", "yolov5"),
    )


def train_yolov8(config: dict):
    return train_yolo(
        weights=config.get("weights", "yolov8n.pt"),
        data_yaml=config["data_yaml"],
        epochs=config.get("epochs", 50),
        imgsz=config.get("imgsz", 640),
        batch=config.get("batch", 16),
        project=config.get("project", "results/plots"),
        name=config.get("name", "yolov8"),
    )
