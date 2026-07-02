"""
Torchvision-детекторы: Faster R-CNN, SSD, RetinaNet.

Все модели берутся предобученными на COCO, после чего голова классификации
заменяется под наше число классов (fine-tuning). num_classes ВКЛЮЧАЕТ фон,
поэтому = len(classes) + 1.
"""
from __future__ import annotations

from functools import partial

import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    retinanet_resnet50_fpn,
    ssd300_vgg16,
)
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.ssd import SSDClassificationHead


def build_faster_rcnn(num_classes: int) -> torch.nn.Module:
    """Faster R-CNN (ResNet-50 + FPN) — двухстадийный детектор."""
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def build_retinanet(num_classes: int) -> torch.nn.Module:
    """RetinaNet (ResNet-50 + FPN) — одностадийный с focal loss."""
    model = retinanet_resnet50_fpn(weights="DEFAULT")
    num_anchors = model.head.classification_head.num_anchors
    in_channels = model.backbone.out_channels
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=partial(torch.nn.GroupNorm, 32),
    )
    return model


def build_ssd(num_classes: int) -> torch.nn.Module:
    """SSD300 (VGG-16) — одностадийный anchor-based детектор."""
    model = ssd300_vgg16(weights="DEFAULT")
    in_channels = det_utils.retrieve_out_channels(model.backbone, (300, 300))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
    )
    return model

TORCHVISION_BUILDERS = {
    "faster_rcnn": build_faster_rcnn,
    "retinanet": build_retinanet,
    "ssd": build_ssd,
}
