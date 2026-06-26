"""
augment_dataset.py — аугментация train-сплита датасета дорожных знаков.

Работает поверх вашей текущей структуры:

    src/dataset/processed/
        train/
            images/   *.ppm / *.jpg / *.png ...
            labels/   *.txt  (формат YOLO: class x_center y_center width height,
                              нормализованные значения, индексация классов с 0)

Важно:
  - Аугментируется ТОЛЬКО train. val не трогаем — иначе метрики моделей друг
    с другом будет нечестно сравнивать (искусственно "размноженные" картинки
    в val искажают оценку качества).
  - Исходные файлы НЕ перезаписываются и НЕ изменяются — для каждой картинки
    создаются только новые файлы с суффиксом "_augN", рядом с оригиналами,
    в тех же папках images/ и labels/. Так Ultralytics YOLO подхватит их
    автоматически (он сканирует все файлы в images/train), без перенастройки.
  - Ресайз к единому размеру здесь намеренно не делается: YOLO (imgsz),
    DETR (DetrImageProcessor) и Faster R-CNN/SSD сами приводят изображения
    к нужному им входному размеру на этапе обучения — дублировать это
    в препроцессинге не нужно и может только потерять качество.
  - После запуска этого скрипта нужно ПЕРЕСОЗДАТЬ annotations_coco.json
    (через yolo_to_coco.py), чтобы новые "_augN" картинки попали и в COCO-
    аннотации для Faster R-CNN / SSD / EfficientDet / DETR.

Запуск:
    python -m src.utils.augment_dataset
"""

import argparse
from pathlib import Path

import albumentations as A
import cv2
from tqdm import tqdm

# =====================================================================================
# Настройки
# =====================================================================================
DEFAULT_DATASET_ROOT = Path("src/dataset/processed")

MULTIPLIER = 3        # сколько аугментированных копий генерировать из ОДНОЙ картинки
MIN_BOX_PIXELS = 4    # минимальный размер бокса в пикселях после аугментации;
                       # боксы меньше отбрасываются (защита от вырожденных боксов,
                       # которые приводили к NaN при обучении DETR ранее в этом проекте)

# =====================================================================================
# Пайплайн аугментаций (Albumentations работает прямо в формате YOLO — нормализованные
# координаты на вход и на выход, поэтому здесь не нужна ручная денормализация боксов)
# =====================================================================================
transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.05, 0.05),
            rotate=(-10, 10),
            border_mode=cv2.BORDER_CONSTANT,
            fill=(114, 114, 114),
            p=0.5,
        ),
        A.GaussNoise(p=0.2),
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3),
)


def read_yolo_label(label_path: Path):
    """Читает .txt-файл YOLO-разметки -> (список боксов [cx,cy,w,h], список классов)."""
    bboxes, class_labels = [], []
    if not label_path.exists():
        return bboxes, class_labels

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id, cx, cy, w, h = parts
            bboxes.append([float(cx), float(cy), float(w), float(h)])
            class_labels.append(int(float(cls_id)))

    return bboxes, class_labels


def write_yolo_label(label_path: Path, bboxes, class_labels):
    with open(label_path, "w", encoding="utf-8") as f:
        for (cx, cy, w, h), cls_id in zip(bboxes, class_labels):
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Аугментация train-сплита YOLO-датасета")
    parser.add_argument(
        "--dataset_root",
        default=str(DEFAULT_DATASET_ROOT),
        help="Путь к корню обработанного датасета (по умолчанию: src/dataset/processed)",
    )
    parser.add_argument(
        "--multiplier",
        type=int,
        default=MULTIPLIER,
        help="Сколько аугментированных копий генерировать из одной картинки",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    train_images_dir = dataset_root / "train" / "images"
    train_labels_dir = dataset_root / "train" / "labels"
    classes_file = dataset_root / "classes.txt"

    if classes_file.exists():
        with open(classes_file, "r", encoding="utf-8") as f:
            num_classes = sum(1 for line in f if line.strip())
        print(f"classes.txt найден, классов: {num_classes}")
    else:
        print("[WARN] classes.txt не найден — пропускаю проверку количества классов")

    if not train_images_dir.exists():
        print(f"[ERROR] Папка с изображениями не найдена: {train_images_dir}")
        return
    train_labels_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".ppm", ".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted(
        p for p in train_images_dir.iterdir()
        if p.suffix.lower() in image_extensions and "_aug" not in p.stem
    )

    if not image_paths:
        print(f"[ERROR] Не найдено изображений в {train_images_dir}")
        return

    print(f"Найдено исходных изображений: {len(image_paths)}")
    print(f"Будет сгенерировано копий на каждую: {args.multiplier}")

    generated = 0
    skipped_empty = 0

    for img_path in tqdm(image_paths, desc="Аугментация"):
        label_path = train_labels_dir / (img_path.stem + ".txt")
        bboxes, class_labels = read_yolo_label(label_path)

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[WARN] Не удалось открыть {img_path}, пропускаю")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        for i in range(1, args.multiplier + 1):
            try:
                result = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            except Exception as e:
                print(f"[WARN] Ошибка аугментации {img_path.name} (копия {i}): {e}")
                continue

            aug_image = result["image"]
            aug_bboxes = result["bboxes"]
            aug_classes = result["class_labels"]

            # Защита от вырожденных боксов: min_visibility у Albumentations проверяет
            # ДОЛЮ видимой площади бокса, а не его абсолютный размер в пикселях.
            # После поворота/скейла бокс может остаться формально "видимым", но
            # сжаться до 1 пикселя — отбрасываем такие отдельно.
            filtered_bboxes, filtered_classes = [], []
            for (cx, cy, w, h), cls_id in zip(aug_bboxes, aug_classes):
                box_w_px = w * width
                box_h_px = h * height

                if box_w_px < MIN_BOX_PIXELS or box_h_px < MIN_BOX_PIXELS:
                    continue  # слишком маленький бокс

                aspect = max(box_w_px / (box_h_px + 1e-6), box_h_px / (box_w_px + 1e-6))
                if aspect > 8.0:
                    continue  # слишком вытянутый бокс

                filtered_bboxes.append([cx, cy, w, h])
                filtered_classes.append(int(cls_id))

            if not filtered_bboxes:
                skipped_empty += 1
                continue

            new_stem = f"{img_path.stem}_aug{i}"
            new_img_path = train_images_dir / f"{new_stem}{img_path.suffix}"
            new_label_path = train_labels_dir / f"{new_stem}.txt"

            save_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(new_img_path), save_bgr)
            write_yolo_label(new_label_path, filtered_bboxes, filtered_classes)

            generated += 1

    print(f"\nГотово! Сгенерировано аугментированных пар (картинка+разметка): {generated}")
    print(f"Пропущено (все боксы стали вырожденными после аугментации): {skipped_empty}")
    print("\n[ВАЖНО] Теперь пересоздайте annotations_coco.json для train, "
          "запустив yolo_to_coco.py — иначе новые '_augN' картинки не попадут "
          "в COCO-аннотации для Faster R-CNN / SSD / EfficientDet / DETR.")


if __name__ == "__main__":
    main()
