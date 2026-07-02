"""Точка входа проекта детекции дорожных знаков (GTSDB).

Режимы:
  prepare  — конвертировать уже скачанный GTSDB (data/raw/) в YOLO + JSON-манифест
  train    — обучить модель(и):  --model yolov8 | ssd | ... | all
  eval     — оценить и сравнить обученные модели, построить графики/таблицы
  predict  — инференс на фото: --model <name> --source <path>

Перед --mode prepare датасет GTSDB (изображения + gt.txt) должен уже
лежать в data/raw/ — сам режим ничего не скачивает.

Примеры:
  python main.py --mode prepare
  python main.py --mode train --model all
  python main.py --mode train --model yolov8 --epochs 50 --batch 8
  python main.py --mode eval  --model all
  python main.py --mode predict --model yolov8 --source demo_inputs/sign.jpg
"""
from __future__ import annotations

import argparse

from src.models.registry import ALL_MODELS
from src.utils.utils import get_logger, load_config, set_seed

log = get_logger("main")


def _resolve_models(arg: str) -> list[str]:
    if arg == "all":
        return ALL_MODELS
    models = [m.strip() for m in arg.split(",") if m.strip()]
    for m in models:
        if m not in ALL_MODELS:
            raise SystemExit(f"Неизвестная модель '{m}'. Доступны: all, {ALL_MODELS}")
    return models


def main():
    ap = argparse.ArgumentParser(description="Детекция дорожных знаков (GTSDB): сравнение 5 моделей")
    ap.add_argument("--mode", required=True, choices=["prepare", "train", "eval", "predict"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default="all", help="имя модели, список через запятую или 'all'")
    ap.add_argument("--source", help="путь к фото для режима predict")
    ap.add_argument("--ckpt", help="путь к чекпойнту (необязательно)")
    # частые переопределения гиперпараметров без правки конфига
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--lr", type=float)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.batch:
        cfg["train"]["batch_size"] = args.batch
    if args.lr:
        cfg["train"]["lr"] = args.lr
    set_seed(cfg.get("seed", 42))

    if args.mode == "prepare":
        from src.dataset.prepare import prepare
        prepare(args.config)

    elif args.mode == "train":
        from src.training.train import train_model
        for name in _resolve_models(args.model):
            log.info(">>> Обучение модели: %s", name)
            train_model(name, cfg)

    elif args.mode == "eval":
        from src.evaluation.compare import evaluate_models
        evaluate_models(_resolve_models(args.model), cfg)

    elif args.mode == "predict":
        if not args.source:
            raise SystemExit("Для режима predict укажите --source <фото>")
        from src.utils.inference import run_inference
        models = _resolve_models(args.model)
        if len(models) != 1:
            raise SystemExit("Для predict укажите ровно одну модель через --model")
        run_inference(models[0], args.source, cfg, args.ckpt)


if __name__ == "__main__":
    main()
