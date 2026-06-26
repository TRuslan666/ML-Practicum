#!/usr/bin/env python3
"""
yolo_to_coco.py

Конвертирует датасет в формате YOLO (images/ + labels/ с .txt-аннотациями)
в формат COCO JSON, который требуется для обучения DETR, Faster R-CNN,
SSD и EfficientDet (через HuggingFace transformers / torchvision / timm).

Ожидаемая входная структура:

    dataset/
        train/
            images/   *.ppm (или .jpg, .png — любой формат, читаемый Pillow)
            labels/   *.txt  (формат YOLO: class x_center y_center width height,
                              все значения нормализованы в диапазоне [0, 1])
        val/
            images/
            labels/

Имя .txt-файла должно совпадать с именем картинки (например image001.ppm -> image001.txt).
Если для картинки нет .txt-файла или файл пустой — считается, что на изображении нет объектов
(image включается в COCO json без аннотаций).

Результат:

    dataset/
        train/
            annotations_coco.json
        val/
            annotations_coco.json

Использование:

    python yolo_to_coco.py --dataset_root dataset --classes classes.txt

    где classes.txt — текстовый файл с названиями классов, одно на строку,
    в том порядке, в котором они закодированы как 0, 1, 2, ... в .txt-аннотациях.

    Если файла classes.txt нет, классы будут названы автоматически:
    class_0, class_1, ... (на основе максимального индекса, встреченного в данных).
"""

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image


def load_classes(classes_path: str | None, labels_dirs: list[Path]) -> list[str]:
    """Загружает имена классов из файла, либо генерирует их автоматически."""
    if classes_path and os.path.isfile(classes_path):
        with open(classes_path, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        print(f"[OK] Загружено {len(names)} классов из {classes_path}")
        return names

    # Автогенерация: проходим по всем .txt и находим максимальный индекс класса
    max_id = -1
    for labels_dir in labels_dirs:
        if not labels_dir.exists():
            continue
        for txt_file in labels_dir.glob("*.txt"):
            with open(txt_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    cls_id = int(float(line.split()[0]))
                    max_id = max(max_id, cls_id)

    if max_id < 0:
        print("[WARN] Не найдено ни одной аннотации, классы не определены.")
        return []

    names = [f"class_{i}" for i in range(max_id + 1)]
    print(f"[WARN] Файл с именами классов не передан. "
          f"Сгенерированы автоматические имена для {len(names)} классов: {names}")
    return names


def convert_split(images_dir: Path, labels_dir: Path, class_names: list[str],
                   output_json: Path) -> None:
    """Конвертирует один сплит (train или val) в COCO JSON."""

    if not images_dir.exists():
        print(f"[SKIP] Папка с изображениями не найдена: {images_dir}")
        return

    image_extensions = {".ppm", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in image_extensions
    )

    if not image_paths:
        print(f"[SKIP] В папке {images_dir} не найдено изображений.")
        return

    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i, "name": name, "supercategory": "none"}
            for i, name in enumerate(class_names)
        ],
    }

    ann_id = 1
    skipped_images = 0
    total_boxes = 0

    for img_id, img_path in enumerate(image_paths, start=1):
        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"[ERROR] Не удалось открыть {img_path}: {e}")
            skipped_images += 1
            continue

        coco["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": width,
            "height": height,
        })

        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            # Нет объектов на изображении — это допустимо для COCO
            continue

        with open(label_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    print(f"[WARN] Некорректная строка {line_num} в {label_path}: '{line}' — пропущена")
                    continue

                cls_id, x_c, y_c, w, h = parts
                cls_id = int(float(cls_id))
                x_c, y_c, w, h = map(float, (x_c, y_c, w, h))

                # Денормализация: YOLO хранит относительные координаты центра и размеры,
                # COCO хранит абсолютные [x_min, y_min, width, height] в пикселях.
                box_w = w * width
                box_h = h * height
                x_min = (x_c * width) - (box_w / 2)
                y_min = (y_c * height) - (box_h / 2)

                # Защита от выхода за границы изображения из-за округления
                x_min = max(0.0, x_min)
                y_min = max(0.0, y_min)
                box_w = min(box_w, width - x_min)
                box_h = min(box_h, height - y_min)

                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls_id,
                    "bbox": [round(x_min, 2), round(y_min, 2), round(box_w, 2), round(box_h, 2)],
                    "area": round(box_w * box_h, 2),
                    "iscrowd": 0,
                })
                ann_id += 1
                total_boxes += 1

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print(f"[DONE] {output_json}")
    print(f"       Изображений: {len(coco['images'])} (пропущено из-за ошибок: {skipped_images})")
    print(f"       Боксов: {total_boxes}")


def main():
    parser = argparse.ArgumentParser(description="Конвертация YOLO-аннотаций в COCO JSON")
    parser.add_argument("--dataset_root", required=True,
                         help="Путь к корню датасета (содержит train/ и val/)")
    parser.add_argument("--classes", default=None,
                         help="Путь к файлу с именами классов (одно имя на строку)")
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                         help="Какие сплиты конвертировать (по умолчанию: train val)")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    labels_dirs = [root / split / "labels" for split in args.splits]
    class_names = load_classes(args.classes, labels_dirs)

    for split in args.splits:
        images_dir = root / split / "images"
        labels_dir = root / split / "labels"
        output_json = root / split / "annotations_coco.json"
        print(f"\n--- Обработка сплита: {split} ---")
        convert_split(images_dir, labels_dir, class_names, output_json)


if __name__ == "__main__":
    main()