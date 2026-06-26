"""
find_bad_boxes.py — ищет потенциально проблемные боксы в annotations_coco.json:
слишком маленькие (по площади/стороне) или с экстремальным соотношением сторон.
Отдельно показывает, сколько таких боксов пришло именно из аугментированных
файлов ("_aug" в имени) — это подтвердит/опровергнет гипотезу, что NaN
во время обучения DETR связан именно с аугментацией.

"""
import argparse
import json
from pathlib import Path

DEFAULT_ANNOTATIONS_FILE = Path("src/dataset/processed/train/annotations_coco.json")

MIN_SIDE_PX = 4          # минимальная сторона бокса в пикселях
MAX_ASPECT_RATIO = 8.0   # максимально допустимое соотношение сторон (w/h или h/w)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Поиск подозрительных COCO bbox")
    parser.add_argument(
        "--annotations",
        default=str(DEFAULT_ANNOTATIONS_FILE),
        help="Путь к annotations_coco.json",
    )
    return parser.parse_args()


def main():
    annotations_file = Path(parse_args().annotations)
    with open(annotations_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

    suspicious = []
    for ann in data["annotations"]:
        x, y, w, h = ann["bbox"]
        filename = image_id_to_filename.get(ann["image_id"], "???")

        reasons = []
        if w < MIN_SIDE_PX or h < MIN_SIDE_PX:
            reasons.append(f"маленькая сторона (w={w:.1f}, h={h:.1f})")

        if w > 0 and h > 0:
            aspect = max(w / h, h / w)
            if aspect > MAX_ASPECT_RATIO:
                reasons.append(f"экстремальный aspect ratio ({aspect:.1f})")

        if w <= 0 or h <= 0:
            reasons.append(f"нулевая/отрицательная сторона (w={w}, h={h})")

        if reasons:
            suspicious.append({
                "ann_id": ann["id"],
                "image_id": ann["image_id"],
                "file_name": filename,
                "bbox": ann["bbox"],
                "reasons": reasons,
            })

    print(f"Всего аннотаций: {len(data['annotations'])}")
    print(f"Подозрительных боксов: {len(suspicious)}")

    aug_count = sum(1 for s in suspicious if "_aug" in s["file_name"])
    orig_count = len(suspicious) - aug_count
    print(f"  из них в аугментированных файлах ('_aug' в имени): {aug_count}")
    print(f"  из них в оригинальных файлах: {orig_count}")

    print("\nПримеры (первые 15):")
    for s in suspicious[:15]:
        print(f"  ann_id={s['ann_id']:>6}  {s['file_name']:<30}  bbox={s['bbox']}  -> {', '.join(s['reasons'])}")

    if suspicious:
        out_path = annotations_file.parent / "suspicious_boxes_report.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(suspicious, f, ensure_ascii=False, indent=2)
        print(f"\nПолный список сохранён в {out_path}")


if __name__ == "__main__":
    main()
