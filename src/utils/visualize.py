"""Отрисовка предсказаний и примеров датасета."""
from __future__ import annotations

import colorsys

import cv2
import numpy as np


def _palette(n: int) -> list[tuple[int, int, int]]:
    """n визуально различимых цветов (BGR для OpenCV)."""
    colors = []
    for i in range(max(1, n)):
        h = i / max(1, n)
        r, g, b = colorsys.hsv_to_rgb(h, 0.7, 0.95)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))
    return colors


def draw_detections(
    image_bgr: np.ndarray,
    dets: list[tuple[tuple[float, float, float, float], float, int]],
    class_names: list[str],
    conf_thr: float = 0.0,
) -> np.ndarray:
    """Рисует рамки + подписи 'тип: уверенность'.

    dets: список ((x1, y1, x2, y2), score, class_id).
    """
    img = image_bgr.copy()
    colors = _palette(len(class_names))
    for (x1, y1, x2, y2), score, cls in dets:
        if score < conf_thr:
            continue
        color = colors[cls % len(colors)]
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, color, 2)
        name = class_names[cls] if 0 <= cls < len(class_names) else str(cls)
        label = f"{name} {score:.2f}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (p1[0], p1[1] - th - bl - 3), (p1[0] + tw, p1[1]), color, -1)
        cv2.putText(
            img, label, (p1[0], p1[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return img
