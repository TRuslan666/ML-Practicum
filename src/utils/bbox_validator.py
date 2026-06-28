"""Comprehensive bounding box validation for detection models.

This module provides utilities to validate bbox integrity, detect NaN/Inf issues,
and ensure boxes meet size constraints before they cause loss explosion during training.
"""
from __future__ import annotations

import math
from typing import Any

import torch


class BboxValidator:
    """Validates and sanitizes bounding boxes for Faster R-CNN training."""

    def __init__(self, min_box_size: float = 2.0, verbose: bool = False) -> None:
        """Initialize validator.
        
        Args:
            min_box_size: Minimum valid box width/height in pixels
            verbose: Print detailed validation messages
        """
        self.min_box_size = float(min_box_size)
        self.verbose = verbose
        self.validation_errors: dict[str, int] = {}

    def validate_targets(self, targets: list[dict[str, torch.Tensor]], image_id: int | None = None) -> None:
        """Validate all targets before model forward pass.
        
        Args:
            targets: List of target dicts with 'boxes', 'labels', etc.
            image_id: Optional image ID for error reporting
            
        Raises:
            ValueError: If any validation fails
        """
        for idx, target in enumerate(targets):
            self._validate_single_target(target, image_id=image_id)

    def _validate_single_target(self, target: dict[str, torch.Tensor], image_id: int | None = None) -> None:
        """Validate a single target dict.
        
        Args:
            target: Target dict with 'boxes', 'labels', etc.
            image_id: Optional image ID for error reporting
            
        Raises:
            ValueError: If validation fails
        """
        if "boxes" not in target:
            return

        boxes = target["boxes"]
        img_id_str = f" (image_id={image_id})" if image_id else ""

        # Check 1: Not empty (OK if empty)
        if boxes.numel() == 0:
            return

        # Check 2: Finite values
        if not torch.isfinite(boxes).all():
            bad_boxes = boxes[~torch.isfinite(boxes).any(dim=1)]
            error_msg = f"NaN/Inf in boxes{img_id_str}: {bad_boxes}"
            self.validation_errors["nan_inf"] = self.validation_errors.get("nan_inf", 0) + 1
            raise ValueError(error_msg)

        # Check 3: Shape correctness (Nx4)
        if len(boxes.shape) != 2 or boxes.shape[1] != 4:
            error_msg = f"Invalid boxes shape {boxes.shape}{img_id_str}, expected (N, 4)"
            self.validation_errors["shape"] = self.validation_errors.get("shape", 0) + 1
            raise ValueError(error_msg)

        # Check 4: Box format: x1 < x2 and y1 < y2
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        
        invalid_x = (x2 <= x1).any()
        invalid_y = (y2 <= y1).any()
        
        if invalid_x or invalid_y:
            bad_idx = ((x2 <= x1) | (y2 <= y1)).nonzero(as_tuple=True)[0]
            bad_boxes = boxes[bad_idx]
            error_msg = f"Invalid box format (x2<=x1 or y2<=y1){img_id_str}: {bad_boxes.tolist()}"
            self.validation_errors["invalid_format"] = self.validation_errors.get("invalid_format", 0) + 1
            raise ValueError(error_msg)

        # Check 5: Minimum size constraint
        widths = x2 - x1
        heights = y2 - y1
        too_small = (widths < self.min_box_size) | (heights < self.min_box_size)
        
        if too_small.any():
            bad_idx = too_small.nonzero(as_tuple=True)[0]
            bad_boxes = boxes[bad_idx]
            bad_sizes = torch.stack([widths[bad_idx], heights[bad_idx]], dim=1)
            error_msg = (
                f"Boxes too small (min={self.min_box_size}){img_id_str}: "
                f"{bad_boxes.tolist()}, sizes: {bad_sizes.tolist()}"
            )
            self.validation_errors["too_small"] = self.validation_errors.get("too_small", 0) + 1
            raise ValueError(error_msg)

        # Check 6: Reasonable coordinate ranges (catch outliers)
        max_coord = boxes.max().item()
        if max_coord > 1e6:
            error_msg = f"Unreasonable box coordinates (max={max_coord}){img_id_str}: {boxes.tolist()}"
            self.validation_errors["unreasonable"] = self.validation_errors.get("unreasonable", 0) + 1
            raise ValueError(error_msg)

        if self.verbose:
            print(f"✓ Target validated{img_id_str}: {len(boxes)} boxes, sizes: "
                  f"w=[{widths.min():.1f}, {widths.max():.1f}], "
                  f"h=[{heights.min():.1f}, {heights.max():.1f}]")

    def validate_batch(self, images: list[torch.Tensor], targets: list[dict[str, torch.Tensor]]) -> None:
        """Validate entire batch before forward pass.
        
        Args:
            images: List of image tensors
            targets: List of target dicts
            
        Raises:
            ValueError: If any validation fails
        """
        if len(images) != len(targets):
            raise ValueError(f"Batch size mismatch: {len(images)} images vs {len(targets)} targets")

        for idx, (image, target) in enumerate(zip(images, targets)):
            img_id = target.get("image_id", [torch.tensor(idx)]).item() if "image_id" in target else idx
            
            # Validate image
            if not torch.isfinite(image).all():
                raise ValueError(f"NaN/Inf in image {idx} (image_id={img_id})")
            
            # Validate target
            self._validate_single_target(target, image_id=img_id)

    def get_error_summary(self) -> str:
        """Get summary of validation errors encountered.
        
        Returns:
            String summary of error counts
        """
        if not self.validation_errors:
            return "No validation errors"
        return "Validation errors: " + ", ".join(
            f"{error}: {count}" for error, count in self.validation_errors.items()
        )

    def reset_errors(self) -> None:
        """Reset error counter."""
        self.validation_errors.clear()


def validate_coco_bbox(bbox: list[float], min_size: float = 2.0) -> bool:
    """Validate a single COCO-format bbox [x, y, width, height].
    
    Args:
        bbox: [x, y, width, height] format
        min_size: Minimum valid width/height
        
    Returns:
        True if bbox is valid, False otherwise
    """
    if len(bbox) != 4:
        return False
    
    x, y, width, height = bbox
    
    # Check finite
    if not all(math.isfinite(v) for v in bbox):
        return False
    
    # Check positive size
    if width < min_size or height < min_size:
        return False
    
    # Check non-negative coordinates
    if x < 0 or y < 0:
        return False
    
    return True


def convert_coco_to_corners(bbox: list[float]) -> list[float]:
    """Convert COCO bbox [x, y, w, h] to corner format [x1, y1, x2, y2].
    
    Args:
        bbox: [x, y, width, height]
        
    Returns:
        [x1, y1, x2, y2]
    """
    x, y, width, height = bbox
    return [x, y, x + width, y + height]


def convert_corners_to_coco(bbox: list[float]) -> list[float]:
    """Convert corner format [x1, y1, x2, y2] to COCO bbox [x, y, w, h].
    
    Args:
        bbox: [x1, y1, x2, y2]
        
    Returns:
        [x, y, width, height]
    """
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]
