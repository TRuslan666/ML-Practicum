"""
main.py — единая точка входа: python main.py --model yolov8

Поддерживаемые значения --model: yolov5, yolov8, detr, faster_rcnn, ssd, efficientdet
(faster_rcnn/ssd/efficientdet подключаются по тому же принципу — добавьте
соответствующий модуль в src/models/ и ветку ниже).
"""

import argparse
import os
from pathlib import Path

import yaml

from src.models.yolo import train_yolov5, train_yolov8
from src.models.faster_rcnn import train_faster_rcnn_from_config
PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # data_yaml в конфиге трактуем как путь относительно корня проекта
    # (там, где лежит main.py), а не относительно текущей рабочей директории —
    # иначе запуск из другой папки (например, из IDE с другим cwd) ломается
    # с FileNotFoundError, как это уже произошло.
    if "data_yaml" in config and not os.path.isabs(config["data_yaml"]):
        config["data_yaml"] = str(PROJECT_ROOT / config["data_yaml"])

    return config


def main():
    parser = argparse.ArgumentParser(description="Запуск обучения одной из моделей проекта")
    parser.add_argument(
        "--model", required=True,
        choices=["yolov5", "yolov8", "detr", "faster_rcnn", "ssd", "efficientdet"],
    )
    parser.add_argument("--config", default=None, help="Путь к yaml-конфигу (по умолчанию configs/<model>.yaml)")
    args = parser.parse_args()

    config_path = args.config or str(PROJECT_ROOT / "configs" / f"{args.model}.yaml")
    config = load_config(config_path)

    if args.model == "yolov5":
        train_yolov5(config)
    elif args.model == "yolov8":
        train_yolov8(config)
    elif args.model == "faster_rcnn":
        train_faster_rcnn_from_config(config)
    else:
        raise NotImplementedError(
            f"Модель '{args.model}' пока не подключена — добавьте обёртку в src/models/"
        )


if __name__ == "__main__":
    main()