"""
check_yolo_labels.py — проверяет все .txt в train/labels и val/labels на наличие
индексов классов за пределами диапазона 0..nc-1, заданного в dataset.yaml.

В отличие от check_categories.py (который проверяет annotations_coco.json),
этот скрипт проверяет именно то, что реально читает Ultralytics YOLO —
.txt-файлы, минуя COCO JSON.

Запуск:
    python check_yolo_labels.py
"""

import math
from pathlib import Path

import yaml

DATASET_YAML = Path("data/processed/dataset.yaml")
SPLITS = ["train", "val"]


def main():
    with open(DATASET_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    nc = data["nc"]
    valid_range = set(range(nc))

    print(f"Ожидаемый диапазон классов: 0..{nc - 1} ({nc} классов)\n")

    total_bad = 0

    for split in SPLITS:
        labels_dir = DATASET_YAML.parent / split / "labels"
        if not labels_dir.exists():
            print(f"[SKIP] {labels_dir} не существует")
            continue

        bad_entries = []
        for txt_file in labels_dir.glob("*.txt"):
            with open(txt_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if not parts:
                        continue
                    try:
                        cls_id = int(float(parts[0]))
                    except ValueError:
                        bad_entries.append((txt_file.name, line_num, line, "не число"))
                        continue

                    if cls_id not in valid_range:
                        bad_entries.append((txt_file.name, line_num, line, f"class_id={cls_id} вне диапазона"))

                    # КЛЮЧЕВАЯ ПРОВЕРКА: NaN/Inf в координатах. В Python "nan < X"
                    # всегда False, поэтому обычные проверки на размер/aspect ratio
                    # такие значения не отлавливают — они тихо проходят все фильтры
                    # и могут вызывать illegal memory access на GPU при обучении.
                    if len(parts) == 5:
                        try:
                            coords = [float(p) for p in parts[1:]]
                            if any(math.isnan(c) or math.isinf(c) for c in coords):
                                bad_entries.append((txt_file.name, line_num, line, "NaN/Inf в координатах бокса"))
                        except ValueError:
                            bad_entries.append((txt_file.name, line_num, line, "координата не число"))

        print(f"--- {split} ---")
        print(f"Проверено файлов: {len(list(labels_dir.glob('*.txt')))}")
        print(f"Найдено проблемных строк: {len(bad_entries)}")
        for fname, line_num, line, reason in bad_entries[:20]:
            print(f"  {fname}:{line_num}  '{line}'  -> {reason}")
        if len(bad_entries) > 20:
            print(f"  ... и ещё {len(bad_entries) - 20}")
        print()

        total_bad += len(bad_entries)

    if total_bad == 0:
        print("[OK] Все class_id во всех .txt находятся в допустимом диапазоне.")
    else:
        print(f"[ERROR] Всего найдено {total_bad} проблемных строк — именно это, "
              f"вероятнее всего, и вызывает 'illegal memory access' при обучении YOLO.")


if __name__ == "__main__":
    main()