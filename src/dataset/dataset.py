"""Dataset и DataLoader для torchvision-детекторов (Faster R-CNN, SSD, RetinaNet).

Читает JSON-манифест, подготовленный src/dataset/prepare.py (main.py --mode prepare).
Поддерживает letterbox-изображения 512x512.
"""

from __future__ import annotations

import json
from pathlib import Path

import albumentations as A
import torch
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.utils.utils import get_logger

log = get_logger("dataset")


def build_transforms(cfg: dict, train: bool) -> A.Compose:
    """Аугментации + letterbox для GTSDB"""
    img_size = cfg["data"].get("img_size", 512)
    acfg = cfg.get("augment", {})

    if train:
        tfs = [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(img_size, img_size, border_mode=0, fill=(0, 0, 0)),  # чёрный padding
            A.HorizontalFlip(p=acfg.get("hflip", 0.5)),
            A.RandomBrightnessContrast(p=acfg.get("brightness_contrast", 0.2)),
            A.HueSaturationValue(p=acfg.get("hue_sat", 0.2)),
        ]
        if acfg.get("blur", 0) > 0:
            tfs.append(A.Blur(blur_limit=3, p=acfg["blur"]))
        if acfg.get("scale_rotate", 0) > 0:
            tfs.append(A.ShiftScaleRotate(
                p=acfg["scale_rotate"], 
                rotate_limit=10,
                border_mode=0
            ))
    else:
        tfs = [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(img_size, img_size, border_mode=0, fill=(0, 0, 0)),
        ]

    tfs.append(ToTensorV2())
    return A.Compose(
        tfs,
        bbox_params=A.BboxParams(
            format="pascal_voc",   # [x1, y1, x2, y2]
            label_fields=["labels"],
            min_visibility=0.2,
        ),
    )


class GTSDBDetectionDataset(Dataset):
    """Детекционный датасет для GTSDB."""

    def __init__(self, manifest_path: str | Path, cfg: dict, train: bool = True):
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.items = json.load(f)
        self.tf = build_transforms(cfg, train)
        self.train = train
        log.info(f"Загружено {len(self.items)} изображений ({'train' if train else 'val'})")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        rec = self.items[idx]
        
        image = np.array(Image.open(rec["image"]).convert("RGB"))
        boxes = rec.get("boxes", [])
        labels = rec.get("labels", [])

        # Применяем аугментации
        out = self.tf(image=image, bboxes=boxes, labels=labels)
        
        img_t = out["image"].float() / 255.0   # [0, 1]
        bboxes = out["bboxes"]
        labels = out["labels"]

        if len(bboxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t = torch.as_tensor(bboxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)

        # torchvision требует labels +1 (0 = background)
        if len(labels_t) > 0:
            labels_t = labels_t + 1

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx]),
        }

        return img_t, target


def collate_fn(batch):
    """Коллатор для detection моделей (разные размеры таргетов)."""
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_dataloaders(cfg: dict):
    """Создаёт train и val DataLoader'ы для GTSDB."""
    processed = Path(cfg["data"]["processed"]) / "gtsdb"
    
    if not (processed / "manifest_train.json").exists():
        raise FileNotFoundError(
            f"Манифест не найден: {processed / 'manifest_train.json'}\n"
            "Сначала выполни: python main.py --mode prepare"
        )

    train_ds = GTSDBDetectionDataset(
        processed / "manifest_train.json", cfg, train=True
    )
    val_ds = GTSDBDetectionDataset(
        processed / "manifest_val.json", cfg, train=False
    )

    tcfg = cfg["train"]

    train_loader = DataLoader(
        train_ds,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        num_workers=tcfg.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=tcfg.get("num_workers", 4) > 0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=tcfg["batch_size"],
        shuffle=False,
        num_workers=tcfg.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def load_classes(cfg: dict) -> list[str]:
    """Загружает список классов."""
    path = Path(cfg["data"]["processed"]) / "classes.txt"
    if path.exists():
        return path.read_text(encoding="utf-8").splitlines()
    else:
        # Для GTSDB — 43 класса (0-42)
        return [str(i) for i in range(43)]