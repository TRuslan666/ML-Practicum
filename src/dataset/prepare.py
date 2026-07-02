"""Простая подготовка GTSDB"""

import json
from collections import defaultdict, Counter
from pathlib import Path

from PIL import Image

from src.utils.utils import ensure_dir, get_logger

log = get_logger("prepare")


def letterbox_image(image: Image.Image, target_size: int = 512):
    iw, ih = image.size
    scale = min(target_size / iw, target_size / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = image.resize((nw, nh), Image.Resampling.BICUBIC)
    new_image = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    new_image.paste(image, ((target_size - nw) // 2, (target_size - nh) // 2))
    return new_image, scale, (target_size - nw) // 2, (target_size - nh) // 2


def convert_to_yolo(x1, y1, x2, y2, target_size):
    """Считает YOLO координаты относительно уже измененного (512x512) изображения"""
    xc = (x1 + x2) / 2 / target_size
    yc = (y1 + y2) / 2 / target_size
    bw = (x2 - x1) / target_size
    bh = (y2 - y1) / target_size
    return f"{xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def prepare(config_path=None):
    raw_root = Path("data/raw")
    out_root = ensure_dir(Path("data/processed/gtsdb"))
    gt_txt_path = raw_root / "gt.txt"

    if not gt_txt_path.exists():
        raise FileNotFoundError(f"gt.txt не найден: {gt_txt_path}")

    log.info(f"GTSDB root: {raw_root.resolve()}")

    # Собираем изображения из data/raw/ и всех подпапок
    image_files = []
    for ext in ["*.ppm", "*.PPM", "*.jpg", "*.JPG", "*.png", "*.PNG"]:
        image_files.extend(raw_root.glob(ext))
        image_files.extend(raw_root.glob(f"**/{ext}"))

    image_files = list(dict.fromkeys([p for p in image_files if p.is_file()]))  # уникальные

    log.info(f"Найдено изображений всего: {len(image_files)}")

    # Читаем аннотации
    annotations = defaultdict(list)
    with open(gt_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.split(';')]
            if len(parts) < 6:
                continue
            fname = parts[0]
            box = list(map(int, parts[1:5]))
            cls = int(parts[5])
            annotations[fname].append((*box, cls))

    # Оставляем только изображения с аннотациями
    annotated_files = set(annotations.keys())
    image_files = [f for f in image_files if f.name in annotated_files]

    log.info(f"Из них с аннотациями: {len(image_files)}")

    if len(image_files) == 0:
        raise ValueError("Не найдено изображений с аннотациями!")

    # Разбиение
    image_files.sort()
    split_idx = int(len(image_files) * 0.8)
    train_imgs = image_files[:split_idx]
    val_imgs = image_files[split_idx:]

    log.info(f"Train: {len(train_imgs)}, Val: {len(val_imgs)}")

    # YOLO структура
    yolo_root = ensure_dir(out_root / "yolo")
    for d in ["images/train", "images/val", "labels/train", "labels/val"]:
        ensure_dir(yolo_root / d)

    manifest = {"train": [], "val": []}
    stats = {"train": Counter(), "val": Counter()}
    TARGET_SIZE = 512

    for split_name, imgs in [("train", train_imgs), ("val", val_imgs)]:
        count = 0
        for img_path in imgs:
            fname = img_path.name

            with Image.open(img_path) as im:
                processed_img, scale, pad_x, pad_y = letterbox_image(im, TARGET_SIZE)

            yolo_lines = []
            boxes_json = []
            labels_json = []

            for xmin, ymin, xmax, ymax, cls_id in annotations[fname]:
                # Новые координаты на картинке 512x512
                x1 = int(xmin * scale) + pad_x
                y1 = int(ymin * scale) + pad_y
                x2 = int(xmax * scale) + pad_x
                y2 = int(ymax * scale) + pad_y

                x1 = max(0, min(x1, TARGET_SIZE))
                y1 = max(0, min(y1, TARGET_SIZE))
                x2 = max(0, min(x2, TARGET_SIZE))
                y2 = max(0, min(y2, TARGET_SIZE))

                if (x2 - x1) >= 2 and (y2 - y1) >= 2:
                    # Считаем YOLO-разметку от новой картинки 512х512
                    yolo_str = convert_to_yolo(x1, y1, x2, y2, TARGET_SIZE)
                    yolo_lines.append(f"{cls_id} {yolo_str}")
                    
                    boxes_json.append([x1, y1, x2, y2])
                    labels_json.append(cls_id)
                    stats[split_name][cls_id] += 1

            if not yolo_lines:
                continue

            dst_img = yolo_root / f"images/{split_name}/{fname}"
            dst_img = dst_img.with_suffix('.jpg')
            processed_img.save(dst_img, quality=95)

            (yolo_root / f"labels/{split_name}/{Path(fname).stem}.txt").write_text(
                "\n".join(yolo_lines), encoding="utf-8"
            )

            # Записываем ПОЛНЫЙ путь, чтобы DataLoader не ругался на FileNotFoundError
            abs_image_path = dst_img.resolve()

            manifest[split_name].append({
                "image": str(abs_image_path),
                "width": TARGET_SIZE,
                "height": TARGET_SIZE,
                "boxes": boxes_json,
                "labels": labels_json,
            })
            count += 1

        log.info(f"{split_name}: {count} изображений")

    # Сохранение
    for split in ["train", "val"]:
        with open(out_root / f"manifest_{split}.json", "w", encoding="utf-8") as f:
            json.dump(manifest[split], f, indent=2)

    (out_root / "classes.txt").write_text("\n".join(str(i) for i in range(43)))

    log.info("✅ GTSDB успешно подготовлен!")
    log.info(f"Папка: {out_root.resolve()}")


if __name__ == "__main__":
    prepare()