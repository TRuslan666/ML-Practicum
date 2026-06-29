"""Stable SSD training with Metrics, Logging, and Plotting.

Uses PyTorch + TorchVision + TorchMetrics.
"""
from __future__ import annotations

import json
import math
import random
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import SSD300_VGG16_Weights, ssd300_vgg16
from torchvision.ops import clip_boxes_to_image
from torchvision.transforms import functional as F

# Импортируем официальную метрику COCO mAP
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from src.utils.bbox_validator import BboxValidator

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".pgm", ".tif", ".tiff"}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[2] / path


class CocoDetectionDataset(Dataset):
    """Minimal COCO dataset reader with aggressive target sanitization."""

    def __init__(self, images_dir: str | Path, annotations_file: str | Path, min_box_size: float = 2.0) -> None:
        self.images_dir = _resolve_path(images_dir)
        self.annotations_file = _resolve_path(annotations_file)
        self.min_box_size = float(min_box_size)

        with self.annotations_file.open("r", encoding="utf-8") as file:
            coco = json.load(file)

        self.categories = sorted(coco.get("categories", []), key=lambda item: item["id"])
        self.category_to_label = {category["id"]: idx + 1 for idx, category in enumerate(self.categories)}
        self.label_to_name = {idx + 1: category["name"] for idx, category in enumerate(self.categories)}

        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in coco.get("annotations", []):
            annotations_by_image[annotation["image_id"]].append(annotation)

        self.images: list[dict[str, Any]] = []
        for image in coco.get("images", []):
            image_path = self.images_dir / image["file_name"]
            if image_path.suffix.lower() in SUPPORTED_IMAGE_EXTS and image_path.exists():
                self.images.append({**image, "path": image_path, "annotations": annotations_by_image[image["id"]]})

        if not self.images:
            raise FileNotFoundError(f"No images from {self.annotations_file} were found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_info = self.images[index]
        image = Image.open(image_info["path"]).convert("RGB")
        image_tensor = F.to_tensor(image)
        height, width = image_tensor.shape[-2:]

        boxes: list[list[float]] = []
        labels: list[int] = []
        areas: list[float] = []
        iscrowd: list[int] = []

        for annotation in image_info["annotations"]:
            bbox = annotation.get("bbox", [])
            if len(bbox) != 4 or not all(math.isfinite(float(value)) for value in bbox):
                continue
            x, y, box_width, box_height = map(float, bbox)
            if box_width < self.min_box_size or box_height < self.min_box_size:
                continue
            label = self.category_to_label.get(annotation.get("category_id"))
            if label is None:
                continue

            box = torch.tensor([[x, y, x + box_width, y + box_height]], dtype=torch.float32)
            box = clip_boxes_to_image(box, (height, width))[0]
            clipped_width = float(box[2] - box[0])
            clipped_height = float(box[3] - box[1])
            if clipped_width < self.min_box_size or clipped_height < self.min_box_size:
                continue

            boxes.append(box.tolist())
            labels.append(label)
            areas.append(clipped_width * clipped_height)
            iscrowd.append(int(annotation.get("iscrowd", 0)))

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([int(image_info["id"])]),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }
        return image_tensor, target


def _collate_fn(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


def _build_model(num_classes: int, image_size: int = 300) -> torch.nn.Module:
    weights = SSD300_VGG16_Weights.DEFAULT
    model = ssd300_vgg16(weights=weights, num_classes=num_classes, classifications_per_img=100)
    model.transform.min_size = (image_size,)
    model.transform.max_size = max(image_size, 300)
    return model


def _plot_history(history: dict[str, list[float]], output_dir: Path) -> None:
    """Генерирует и сохраняет графики обучения."""
    epochs = range(1, len(history["train_loss"]) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # График лоссов
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "-o", label="Total Train Loss")
    plt.plot(epochs, history["cls_loss"], label="Class Loss", alpha=0.7)
    plt.plot(epochs, history["box_loss"], label="Box Loss", alpha=0.7)
    plt.title("Training Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    
    # График метрики mAP
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_map"], "-o", color="green", label="Val mAP @50:95")
    plt.plot(epochs, history["val_map_50"], "--", color="lime", label="Val mAP @50")
    plt.title("Validation Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "learning_curves.png", dpi=150)
    plt.close()


def train_ssd_from_config(config: dict[str, Any]) -> torch.nn.Module:
    """Train SSD with logging, mAP calculation, and plot generation."""
    seed = int(config.get("seed", 42))
    _seed_everything(seed)

    output_dir = _resolve_path(config.get("project", "results/")) / config.get("name", "ssd_vgg16")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- НАСТРОЙКА ЛОГИРОВАНИЯ В ФАЙЛ И КОНСОЛЬ ---
    logger = logging.getLogger("SSD_Training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear() # Очистка старых хендлеров, если скрипт перезапускался
    
    file_handler = logging.FileHandler(output_dir / "train_log.txt", encoding="utf-8")
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    device_name = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    image_size = int(config.get("imgsz", 300))
    min_box_size = float(config.get("min_box_size", 2.0))
    
    train_dataset = CocoDetectionDataset(config["train_images"], config["train_annotations"], min_box_size=min_box_size)
    val_dataset = CocoDetectionDataset(config["val_images"], config["val_annotations"], min_box_size=min_box_size)

    num_classes = int(config.get("num_classes", len(train_dataset.categories))) + 1
    model = _build_model(num_classes=num_classes, image_size=image_size).to(device)

    batch_size = int(config.get("batch", 4))  
    workers = int(config.get("dataloader_num_workers", 0))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=workers, collate_fn=_collate_fn)

    base_lr = float(config.get("lr", 1e-3))
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=base_lr,
        momentum=0.9,
        weight_decay=float(config.get("weight_decay", 5e-4))
    )

    warmup_steps = int(config.get("warmup_steps", 200))

    def _apply_warmup_lr(global_step: int) -> None:
        if global_step > warmup_steps:
            return
        if global_step == warmup_steps:
            for param_group in optimizer.param_groups:
                param_group["lr"] = base_lr
            return
        warmup_factor = (global_step + 1) / warmup_steps
        for param_group in optimizer.param_groups:
            param_group["lr"] = base_lr * warmup_factor

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(config.get("lr_step_size", 15)), gamma=0.5)
    grad_clip_norm = float(config.get("grad_clip_norm", 1.0))
    epochs = int(config.get("epochs", 50))

    validator = BboxValidator(min_box_size=min_box_size, verbose=bool(config.get("verbose_validation", False)))

    logger.info(f"Device: {device}; Train images: {len(train_dataset)}; Val images: {len(val_dataset)}")
    logger.info(f"SSD classes: {num_classes}; Batch size={batch_size}; Image size={image_size}")

    best_map = -1.0
    global_step = 0
    MAX_VALID_LOSS = 20.0  

    # Словарь для хранения истории обучения (для графиков)
    history = {
        "train_loss": [],
        "cls_loss": [],
        "box_loss": [],
        "val_map": [],
        "val_map_50": []
    }

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        skipped_batches = 0
        total_batches = len(train_loader)
        cls_losses = []
        box_losses = []
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            try:
                validator.validate_batch(images, targets)
            except ValueError as error:
                skipped_batches += 1
                logger.warning(f"[SKIP DATA] Epoch={epoch} Batch={batch_idx}: {error}")
                continue

            _apply_warmup_lr(global_step)
            global_step += 1

            images = [image.to(device, non_blocking=True) for image in images]
            targets = [{key: value.to(device, non_blocking=True) for key, value in target.items()} for target in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            if not torch.isfinite(losses) or losses.item() > MAX_VALID_LOSS:
                details = {name: float(value.detach().cpu()) for name, value in loss_dict.items()}
                logger.error(f"[SKIP GRAD] Epoch={epoch} Batch={batch_idx}: Аномальный лосс ({losses.item():.4f})! Детали: {details}")
                skipped_batches += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            cls_losses.append(loss_dict["classification"].item())
            box_losses.append(loss_dict["bbox_regression"].item())

            optimizer.zero_grad(set_to_none=True)
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            
            epoch_losses.append(losses.item())

        if total_batches > 0 and skipped_batches == total_batches:
            logger.critical(f"Эпоха {epoch}: ВСЕ батчи упали или отфильтрованы! Остановка.")
            raise RuntimeError(f"Эпоха {epoch}: Обучение сорвано взрывом градиентов.")

        scheduler.step()
        
        # Считаем средние лоссы за эпоху
        train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("nan")
        avg_cls = sum(cls_losses) / len(cls_losses) if cls_losses else 0.0
        avg_box = sum(box_losses) / len(box_losses) if box_losses else 0.0

        # --- ВАЛИДАЦИЯ И РАСЧЕТ mAP ЧЕРЕЗ TORCHMETRICS ---
        model.eval()
        # Инициализируем метрику COCO mAP (подсчет mAP@50:95 и mAP@50)
        metric_coco = MeanAveragePrecision(box_format="xyxy")
        
        logger.info(f"Запуск валидации для Эпохи {epoch:03d}...")
        with torch.no_grad():
            for images, targets in val_loader:
                images_dev = [img.to(device) for img in images]
                outputs = model(images_dev)
                
                # Переносим предсказания и таргеты на CPU в формате, который ждет torchmetrics
                preds = [{k: v.cpu() for k, v in out.items()} for out in outputs]
                targets_cpu = [{k: v.cpu() for k, v in tg.items()} for tg in targets]
                
                metric_coco.update(preds, targets_cpu)
        
        # Вычисляем финальные метрики валидации
        metrics_results = metric_coco.compute()
        val_map = float(metrics_results["map"].item())
        val_map_50 = float(metrics_results["map_50"].item())

        # Логируем итоги эпохи
        logger.info(
            f"Epoch {epoch:03d}/{epochs} Закончена. "
            f"Train Loss: {train_loss:.4f} [Cls: {avg_cls:.4f}, Box: {avg_box:.4f}] | "
            f"Val mAP@50:95: {val_map:.4f} | Val mAP@50: {val_map_50:.4f} | "
            f"Пропущено батчей: {skipped_batches}/{total_batches}"
        )

        # Сохраняем значения в историю
        history["train_loss"].append(train_loss)
        history["cls_loss"].append(avg_cls)
        history["box_loss"].append(avg_box)
        history["val_map"].append(val_map)
        history["val_map_50"].append(val_map_50)

        # Перестраиваем графики каждую эпоху
        _plot_history(history, output_dir)

        # Сохранение чекпоинтов
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": {"map": val_map, "map_50": val_map_50},
            "config": config,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        
        # Сохраняем модель по лучшей метрике mAP, а не по лоссу!
        if val_map > best_map:
            best_map = val_map
            torch.save(checkpoint, output_dir / "best.pt")
            logger.info(f"--> Найдена лучшая модель на эпохе {epoch} с mAP: {best_map:.4f}! Сохранено в best.pt")

    logger.info(f"Обучение успешно завершено. Все результаты сохранены в: {output_dir}")
    return model