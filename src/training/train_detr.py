import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from util_detr import PlotMetricsCallback
import json
from functools import partial
import logging


import torch
import torchvision
import yaml
from transformers import (
    DetrConfig,
    DetrForObjectDetection,
    DetrImageProcessor,
    Trainer,
    TrainingArguments,
    utils as tf_utils
)


# =====================================================================================
# ВАЖНО: классы и функции, которые DataLoader должен "распаковывать" (pickle) в
# worker-процессах, определены на уровне модуля — НЕ внутри if __name__ == '__main__'.
# На Windows multiprocessing использует spawn: каждый worker заново импортирует этот
# файл как __mp_main__ и должен суметь найти эти определения. Если поместить их внутрь
# блока main, получите AttributeError: Can't get attribute '...' on '__mp_main__'.
# =====================================================================================
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# 1. Создаем директорию для логов, если её нет
log_dir = "results/logs"
os.makedirs(log_dir, exist_ok=True)

# Имя файла лога (можно добавить дату/время, если нужно)
log_file_path = os.path.join(log_dir, "training.log")

# 2. Настраиваем корневой логгер Python
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Очистим старые хендлеры, чтобы не было дублирования в консоли
if logger.hasHandlers():
    logger.handlers.clear()

# Хендлер для записи в файл
file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

# Хендлер для вывода в консоль (чтобы вы видели прогресс в терминале)
console_handler = logging.StreamHandler()
console_formatter = logging.Formatter('%(message)s') # Для Trainer лучше оставить чистый вывод
console_handler.setFormatter(console_formatter)
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

# 3. Настраиваем логирование библиотеки Transformers, чтобы она использовала наш логгер
tf_utils.logging.enable_default_handler()
tf_utils.logging.enable_explicit_format()
# Перенаправляем логи transformers в стандартный logging
# (Обычно Hugging Face делает это автоматически, если настроен корневой логгер)


class DetrDataset(torch.utils.data.Dataset):
    """Обёртка над torchvision.datasets.CocoDetection под формат HuggingFace DETR."""

    def __init__(self, images_dir, annotation_file, processor):
        self.dataset = torchvision.datasets.CocoDetection(
            root=images_dir,
            annFile=annotation_file,
        )
        self.processor = processor
        
        # --- НОВЫЙ БЛОК: Создаем маппинг старых ID в непрерывные 0..N-1 ---
        # Собираем все уникальные id категорий из COCO
        self.coco_categories = sorted(self.dataset.coco.getCatIds())
        # Карта: {старый_id: новый_индекс_от_0_до_N}
        self.old2new_id = {old_id: idx for idx, old_id in enumerate(self.coco_categories)}
        # -----------------------------------------------------------------

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, annotations = self.dataset[idx]
        image_id = self.dataset.ids[idx]

        clean_annotations = []
        for ann in annotations:
            _, _, w, h = ann["bbox"]
            if w <= 1e-3 or h <= 1e-3:
                continue
            
            # Глубокое копирование, чтобы не испортить исходный self.dataset
            ann_copy = ann.copy()
            # ПЕРЕМАПЛИВАЕМ ID КЛАССА В НЕПРЕРЫВНЫЙ ДИАПАЗОН
            ann_copy["category_id"] = self.old2new_id[ann["category_id"]]
            
            clean_annotations.append(ann_copy)

        target = {
            "image_id": image_id,
            "annotations": clean_annotations,
        }

        encoding = self.processor(
            images=image,
            annotations=target,
            return_tensors="pt",
        )

        pixel_values = encoding["pixel_values"].squeeze()
        labels = encoding["labels"][0]

        return pixel_values, labels


def collate_fn(batch, processor=None):
    """
    Ручной паддинг батча до общего размера + создание pixel_mask.

    Не используем processor.pad(...), так как его сигнатура нестабильна между
    версиями transformers (например, аргумент return_tensors в какой-то момент
    был убран/переименован) — паддинг на чистом PyTorch не зависит от версии.

    Параметр processor оставлен для совместимости с functools.partial(collate_fn,
    processor=processor), но не используется внутри.
    """
    pixel_values = [item[0] for item in batch]  # каждый: тензор [C, H, W], размеры могут отличаться
    labels = [item[1] for item in batch]

    max_h = max(pv.shape[-2] for pv in pixel_values)
    max_w = max(pv.shape[-1] for pv in pixel_values)

    padded_pixel_values = []
    pixel_masks = []
    for pv in pixel_values:
        c, h, w = pv.shape

        padded = torch.zeros((c, max_h, max_w), dtype=pv.dtype)
        padded[:, :h, :w] = pv
        padded_pixel_values.append(padded)

        # pixel_mask: 1 — реальные пиксели, 0 — добавленный паддинг
        mask = torch.zeros((max_h, max_w), dtype=torch.long)
        mask[:h, :w] = 1
        pixel_masks.append(mask)

    return {
        "pixel_values": torch.stack(padded_pixel_values),
        "pixel_mask": torch.stack(pixel_masks),
        "labels": labels,
    }


def load_class_names(yaml_path: str) -> dict[int, str]:
    """Загружает реальные названия классов из dataset.yaml (формат YOLO: nc + names)."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data["names"]  # dict {0: "speed limit 20", 1: "speed limit 30", ...}
    # YAML может прочитать ключи как int или str — нормализуем к int
    return {int(k): v for k, v in names.items()}


def build_id2label(coco_json_path: str, yaml_path: str) -> dict[int, str]:
    """
    Строит id2label, используя category_id из реального COCO-датасета (а не
    предполагая, что они всегда 0..N-1 подряд) и подставляя настоящие названия
    знаков из dataset.yaml вместо плейсхолдеров "class_N".
    """
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    real_names = load_class_names(yaml_path)

    id2label = {}
    for cat in coco_data["categories"]:
        cat_id = cat["id"]
        # Если в yaml есть название для этого id — берём его, иначе оставляем как есть
        id2label[cat_id] = real_names.get(cat_id, cat["name"])

    return id2label


def main():
    # -----------------------------------------------------------------------
    # 1. Устройство
    # -----------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Обучение будет проходить на: {device}")

    # -----------------------------------------------------------------------
    # 2. Пути (поправьте под свою структуру, если отличается)
    # -----------------------------------------------------------------------
    train_images_dir = "src/dataset/processed/train/images"
    train_ann_file = "src/dataset/processed/train/annotations_coco.json"
    val_images_dir = "src/dataset/processed/val/images"
    val_ann_file = "src/dataset/processed/val/annotations_coco.json"
    yaml_path = "src/dataset/dataset.yaml"

    # -----------------------------------------------------------------------
    # 3. Реальные названия классов (а не "class_0", "class_1"...)
    # -----------------------------------------------------------------------
    id2label = build_id2label(train_ann_file, yaml_path)
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)

    print(f"Найдено классов: {num_labels}")
    print(f"category_id диапазон: {min(id2label.keys())}..{max(id2label.keys())}")

    # Проверка непрерывности индексов 0..N-1 — DETR этого ожидает
    expected_ids = set(range(num_labels))
    actual_ids = set(id2label.keys())
    if actual_ids != expected_ids:
        print("[WARN] category_id не образуют непрерывный диапазон 0..N-1!")
        print(f"       Ожидалось: {sorted(expected_ids)[:5]}...")
        print(f"       Получено:  {sorted(actual_ids)[:5]}...")
        print("       Это может привести к ошибкам индексации при обучении.")

    # -----------------------------------------------------------------------
    # 4. Процессор и модель
    # -----------------------------------------------------------------------
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    config = DetrConfig.from_pretrained("facebook/detr-resnet-50")
    config.use_pretrained_backbone = True  # ImageNet-предобученный ResNet50 бэкбон
    config.num_labels = num_labels
    config.id2label = id2label
    config.label2id = label2id

    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        config=config,
        ignore_mismatched_sizes=True,  # отрезаем голову на 80 классов COCO, ставим свою
    )

    # -----------------------------------------------------------------------
    # 5. Датасеты
    # -----------------------------------------------------------------------
    train_dataset = DetrDataset(train_images_dir, train_ann_file, processor)
    eval_dataset = DetrDataset(val_images_dir, val_ann_file, processor)

    print(f"Train: {len(train_dataset)} изображений")
    print(f"Val:   {len(eval_dataset)} изображений")

    # -----------------------------------------------------------------------
    # 6. Параметры обучения
    #    Настроено под GTX 1660 Super (6 ГБ VRAM, Turing — без поддержки bf16).
    #    fp16/bf16 ОТКЛЮЧЕНЫ: DETR-loss (giou + matcher) нестабилен в half precision
    #    и приводит к NaN после нескольких десятков шагов (проверено на практике).
    # -----------------------------------------------------------------------
    training_args = TrainingArguments(
        logging_dir="results/logs",             # Директория для TensorBoard логов (если используются)
        logging_strategy="steps",               # Логируем по шагам

        output_dir="results/models/detr_finetuned_results",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        learning_rate=1e-05,
        weight_decay=1e-4,
        logging_steps=10,
        remove_unused_columns=False,

        # Валидация на каждой эпохе + сохранение лучшего чекпоинта
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Загрузка данных
        dataloader_num_workers=2,     # на Windows безопасно ставить 2; при проблемах — 0
        dataloader_pin_memory=True,

        # Стабильность (защита от NaN — важно для DETR)
        max_grad_norm=1.0,
        fp16=False,
        bf16=False,
    )

    # -----------------------------------------------------------------------
    # 7. Trainer
    # -----------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=partial(collate_fn, processor=processor),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[PlotMetricsCallback(output_dir="results/plots")]
    )

    # -----------------------------------------------------------------------
    # 8. Обучение
    # -----------------------------------------------------------------------
    print("Старт обучения...")
    trainer.train()

    # -----------------------------------------------------------------------
    # 9. Сохранение
    # -----------------------------------------------------------------------
    model.save_pretrained("src/models/detr/my_best_detr_model")
    processor.save_pretrained("src/models/detr/my_best_detr_model")
    print("Обучение завершено! Модель сохранена в папку 'my_best_detr_model'")


if __name__ == "__main__":
    main()