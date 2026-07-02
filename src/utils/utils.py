"""Вспомогательные функции: воспроизводимость, конфиг, логирование, устройство."""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """Фиксируем все источники случайности для воспроизводимости (п. 3.6)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Детерминизм cudnn немного замедляет, но даёт повторяемость.
    torch.backends.cudnn.benchmark = True


def load_config(path: str | Path) -> dict[str, Any]:
    """Читаем YAML-конфиг эксперимента."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(prefer: str = "cuda") -> torch.device:
    """Возвращаем доступное устройство (GPU при наличии — требование задания)."""
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_logger(name: str = "cv-project", log_file: str | Path | None = None) -> logging.Logger:
    """Логгер в консоль и (опционально) в файл."""
    logger = logging.getLogger(name)
    if logger.handlers:  # уже настроен
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def ensure_dir(path: str | Path) -> Path:
    """Создаём директорию, если её нет."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
