"""
src/inference/predict.py

Универсальный скрипт для запуска инференса (предсказания) на изображениях
для трех архитектур: YOLOv5, YOLOv8 и Faster R-CNN.
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

# Словарь названий классов (замените или импортируйте ваш список из dataset.yaml)
# Для примера взяты первые несколько классов GTSRB
CLASS_NAMES = {
    0: "speed limit 20",
    1: "speed limit 30",
    2: "speed limit 50",
    # ... до 42
}


def draw_predictions(image_path, boxes, scores, class_ids, output_path, conf_threshold=0.3):
    """Отрисовка ограничивающих рамок на изображении с помощью OpenCV."""
    img = cv2.imread(str(image_path))
    h, w, _ = img.shape

    for box, score, cid in zip(boxes, scores, class_ids):
        if score < conf_threshold:
            continue

        # Координаты бокса
        xmin, ymin, xmax, ymax = map(int, box)

        # Текст метки
        label_text = f"{CLASS_NAMES.get(cid, f'Class {cid}')}: {score:.2f}"

        # Рисуем рамку и текст
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(
            img,
            label_text,
            (xmin, max(ymin - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    cv2.imwrite(str(output_path), img)
    print(f"Результат успешно сохранен в: {output_path}")


def predict_yolov8(image_path, weights_path, conf_threshold):
    """Инференс через официальную библиотеку Ultralytics (YOLOv8)."""
    from ultralytics import YOLO

    model = YOLO(weights_path)
    results = model(image_path, conf=conf_threshold)[0]

    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy().astype(int)

    return boxes, scores, class_ids


def predict_yolov5(image_path, weights_path, conf_threshold):
    """Инференс через PyTorch Hub для YOLOv5 (локальный или удаленный)."""
    # Загружаем модель (YOLOv5 автоматически переводит координаты в пиксели)
    model = torch.hub.load(
        "ultralytics/yolov5", "custom", path=weights_path, force_reload=False
    )
    model.conf = conf_threshold

    results = model(image_path)
    # Получаем DataFrame с результатами детекции
    pred_df = results.pandas().xyxy[0]

    boxes = pred_df[["xmin", "ymin", "xmax", "ymax"]].values
    scores = pred_df["confidence"].values
    class_ids = pred_df["class"].values.astype(int)

    return boxes, scores, class_ids


def predict_faster_rcnn(image_path, weights_path, conf_threshold, num_classes=43):
    """Инференс для Faster R-CNN на чистом PyTorch/Torchvision."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Инициализируем точно такую же структуру модели, как при обучении
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

    # Загружаем сохраненный чекпоинт
    checkpoint = torch.load(weights_path, map_location=device)
    # Проверяем, сохранен ли там чистый state_dict или словарь состояния обучения
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # Предобработка картинки
    img_pil = Image.open(image_path).convert("RGB")
    img_tensor = torchvision.transforms.functional.to_tensor(img_pil).to(device)

    with torch.no_grad():
        predictions = model([img_tensor])[0]

    boxes = predictions["boxes"].cpu().numpy()
    scores = predictions["scores"].cpu().numpy()
    # Смещаем классы обратно (-1), так как при обучении Faster R-CNN мы прибавляли 1 для Background
    class_ids = (predictions["labels"].cpu().numpy() - 1).astype(int)

    return boxes, scores, class_ids


def main():
    parser = argparse.ArgumentParser(description="Универсальный инференс для детекции знаков.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["yolov5", "yolov8", "faster_rcnn"],
        help="Тип архитектуры модели",
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Путь к входному изображению"
    )
    parser.add_argument(
        "--weights", type=str, required=True, help="Путь к файлу весов (.pt)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Порог уверенности (confidence threshold)"
    )
    args = parser.parse_argument_group().parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Ошибка: Изображение {args.image} не найдено!")
        return

    print(f"Запуск инференса [{args.model}] для файла: {image_path.name}...")

    # Вызов нужного предиктора в зависимости от переданного аргумента
    if args.model == "yolov8":
        boxes, scores, class_ids = predict_yolov8(image_path, args.weights, args.conf)
    elif args.model == "yolov5":
        boxes, scores, class_ids = predict_yolov5(image_path, args.weights, args.conf)
    elif args.model == "faster_rcnn":
        boxes, scores, class_ids = predict_faster_rcnn(image_path, args.weights, args.conf)

    # Настраиваем директорию сохранения результатов
    out_dir = Path("results/plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"pred_{args.model}_{image_path.name}"

    # Отрисовываем предсказания
    draw_predictions(image_path, boxes, scores, class_ids, output_path, args.conf)


if __name__ == "__main__":
    main()