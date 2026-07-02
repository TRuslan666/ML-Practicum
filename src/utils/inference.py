"""Инференс обученных моделей на фото.

Поддерживает обе семьи:
  - ultralytics (YOLOv8 / YOLOv5): нативно работает на фото и папках.
  - torchvision (Faster R-CNN / SSD / RetinaNet): letterbox-ресайз + обратный
    пересчёт боксов в координаты исходного изображения.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from src.dataset.dataset import load_classes
from src.models.registry import build_torchvision_model, model_family
from src.utils.utils import ensure_dir, get_device, get_logger
from src.utils.visualize import draw_detections

log = get_logger("inference")


# --------------------------------------------------------------------------- #
#  torchvision-инференс с letterbox
# --------------------------------------------------------------------------- #
def _letterbox(img: np.ndarray, size: int):
    """Масштабирует с сохранением пропорций и паддит до size x size."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, scale, left, top


class TorchvisionDetector:
    """Загружает torchvision-детектор из чекпойнта и предсказывает на кадре."""

    def __init__(self, model_name: str, ckpt: str, cfg: dict):
        self.cfg = cfg
        self.device = get_device(cfg["train"]["device"])
        self.classes = load_classes(cfg)
        self.size = cfg["data"]["img_size"]
        self.conf = cfg["eval"]["conf_threshold"]
        model = build_torchvision_model(model_name, len(self.classes) + 1)
        model.load_state_dict(torch.load(ckpt, map_location=self.device))
        self.model = model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, image_bgr: np.ndarray, conf: float | None = None):
        """conf=None -> порог из конфига (для визуализации).
        Для оценки mAP передайте маленький conf (напр. 0.001)."""
        thr = self.conf if conf is None else conf
        canvas, scale, left, top = _letterbox(image_bgr, self.size)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(self.device)
        out = self.model([tensor])[0]

        dets = []
        for box, score, label in zip(out["boxes"], out["scores"], out["labels"]):
            if float(score) < thr:
                continue
            x1, y1, x2, y2 = box.tolist()
            # обратный пересчёт из letterbox в исходные координаты
            x1 = (x1 - left) / scale
            y1 = (y1 - top) / scale
            x2 = (x2 - left) / scale
            y2 = (y2 - top) / scale
            dets.append(((x1, y1, x2, y2), float(score), int(label) - 1))  # -1: убираем фон
        return dets


def _run_torchvision(detector: TorchvisionDetector, source: Path, out_dir: Path):
    frame = cv2.imread(str(source))
    dets = detector.predict(frame)
    out_path = out_dir / f"{source.stem}_pred.jpg"
    cv2.imwrite(str(out_path), draw_detections(frame, dets, detector.classes))
    log.info("Фото обработано (%d объектов) -> %s", len(dets), out_path)


# --------------------------------------------------------------------------- #
#  ultralytics-инференс (нативный)
# --------------------------------------------------------------------------- #
def _run_ultralytics(model_name: str, ckpt: str, source: Path, cfg: dict, out_dir: Path):
    from src.models.ultralytics_models import build_ultralytics

    model = build_ultralytics(model_name) if not ckpt else _load_ultra(model_name, ckpt)
    model.predict(
        source=str(source),
        conf=cfg["eval"]["conf_threshold"],
        save=True,
        project=str(out_dir),
        name=model_name,
        exist_ok=True,
        device=0 if get_device(cfg["train"]["device"]).type == "cuda" else "cpu",
    )
    log.info("ultralytics-инференс сохранён в %s/%s", out_dir, model_name)


def _load_ultra(model_name: str, ckpt: str):
    from ultralytics import YOLO
    return YOLO(ckpt)


# --------------------------------------------------------------------------- #
#  единая точка входа
# --------------------------------------------------------------------------- #
def run_inference(model_name: str, source: str, cfg: dict, ckpt: str | None = None):
    out_dir = ensure_dir(Path("demo_outputs"))
    src = Path(source)
    if not src.exists():
        raise FileNotFoundError(f"Источник не найден: {src}")

    if model_family(model_name) == "ultralytics":
        if not ckpt:
            # канонический чекпойнт; откат на родную папку ultralytics
            cand = Path(cfg["output"]["checkpoints"]) / model_name / "best.pt"
            if not cand.exists():
                cand = Path(cfg["output"]["logs"]) / "ultralytics" / model_name / "weights" / "best.pt"
            ckpt = str(cand)
        if not Path(ckpt).exists():
            raise FileNotFoundError(
                f"Не нашёл обученные веса {model_name}: {ckpt}. Сначала обучите модель."
            )
        _run_ultralytics(model_name, ckpt, src, cfg, out_dir)
    else:
        if not ckpt:
            ckpt = str(Path(cfg["output"]["checkpoints"]) / model_name / "best.pt")
        detector = TorchvisionDetector(model_name, ckpt, cfg)
        _run_torchvision(detector, src, out_dir)
