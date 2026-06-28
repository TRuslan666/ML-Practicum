"""Stable Faster R-CNN training for this project.

The project stores detection labels as COCO JSON files generated from YOLO
labels.  This module intentionally does not depend on the external
``torch_utils``/``datasets`` training template that used to be pasted here: it
uses only PyTorch + TorchVision and is callable from ``main.py`` via
``train_faster_rcnn_from_config``.

Defaults are conservative for a GTX 1660 Super (6 GB VRAM): no AMP, small batch,
low learning rate, gradient clipping and strict box/loss validation to prevent
NaN/Inf from corrupting the model during training.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import clip_boxes_to_image
from torchvision.transforms import functional as F

# .ppm is used by the traffic-sign dataset in this repository. Pillow can read it,
# so no extra conversion step is required.
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".pgm", ".tif", ".tiff"}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism and disabled benchmark make the first CUDA runs less spiky on
    # small 6 GB GPUs and make failures easier to reproduce.
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


def _build_model(num_classes: int, image_size: int) -> torch.nn.Module:
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights, box_detections_per_img=100)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Keep memory use predictable on GTX 1660 Super. TorchVision still preserves
    # aspect ratio internally, but caps the long side at 800 by default.
    model.transform.min_size = (image_size,)
    model.transform.max_size = max(image_size, 800)
    return model


def _assert_finite_targets(targets: list[dict[str, torch.Tensor]]) -> None:
    for target in targets:
        boxes = target["boxes"]
        if not torch.isfinite(boxes).all():
            raise ValueError(f"NaN/Inf in target boxes before forward(): image_id={target['image_id'].item()}")
        if boxes.numel() and ((boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])).any():
            raise ValueError(f"Invalid target box with non-positive size: image_id={target['image_id'].item()}")


def train_faster_rcnn_from_config(config: dict[str, Any]) -> torch.nn.Module:
    """Train Faster R-CNN using ``configs/faster_rcnn.yaml`` style settings."""

    seed = int(config.get("seed", 42))
    _seed_everything(seed)

    device_name = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    image_size = int(config.get("imgsz", 512))
    min_box_size = float(config.get("min_box_size", 2.0))
    train_dataset = CocoDetectionDataset(config["train_images"], config["train_annotations"], min_box_size=min_box_size)
    val_dataset = CocoDetectionDataset(config["val_images"], config["val_annotations"], min_box_size=min_box_size)

    # +1 for background class required by TorchVision detection heads.
    num_classes = int(config.get("num_classes", len(train_dataset.categories))) + 1
    model = _build_model(num_classes=num_classes, image_size=image_size).to(device)

    batch_size = int(config.get("batch", 2))
    workers = int(config.get("dataloader_num_workers", 0))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=workers, collate_fn=_collate_fn)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config.get("lr", 1e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(config.get("lr_step_size", 8)), gamma=0.1)
    grad_clip_norm = float(config.get("grad_clip_norm", 1.0))
    epochs = int(config.get("epochs", 50))

    output_dir = _resolve_path(config.get("project", "results/models")) / config.get("name", "faster_rcnn_experiment")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}; train images: {len(train_dataset)}; val images: {len(val_dataset)}")
    print(f"Faster R-CNN classes including background: {num_classes}; batch={batch_size}; imgsz={image_size}; amp=False")

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for images, targets in train_loader:
            _assert_finite_targets(targets)
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [{key: value.to(device, non_blocking=True) for key, value in target.items()} for target in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            if not torch.isfinite(losses):
                details = {name: float(value.detach().cpu()) for name, value in loss_dict.items()}
                raise FloatingPointError(f"Non-finite Faster R-CNN loss at epoch {epoch}: {details}")

            optimizer.zero_grad(set_to_none=True)
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            epoch_losses.append(float(losses.detach().cpu()))

        scheduler.step()
        train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("nan")
        print(f"Epoch {epoch:03d}/{epochs}: train_loss={train_loss:.5f}")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "num_classes": num_classes,
            "label_to_name": train_dataset.label_to_name,
            "config": config,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(checkpoint, output_dir / "best.pt")

        # Cheap validation smoke pass: catches CUDA/memory/shape issues without a
        # slow COCO mAP dependency in the core training script.
        model.eval()
        with torch.no_grad():
            for images, _targets in val_loader:
                _ = model([image.to(device, non_blocking=True) for image in images])
                break

    print(f"Training finished. Best checkpoint: {output_dir / 'best.pt'}")
    return model
