"""Построение графиков для отчёта (раздел 3.7 задания).

  - кривые обучения (loss и метрики по эпохам);
  - сравнительная столбчатая диаграмма моделей;
  - матрица ошибок (какие типы самолётов путаются);
  - зависимость качества от гиперпараметра.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.utils import ensure_dir

sns.set_theme(style="whitegrid")


def _read_history(model_name: str, logs_dir: Path) -> pd.DataFrame | None:
    """История torchvision (*_history.json) или ultralytics (results.csv)."""
    j = logs_dir / f"{model_name}_history.json"
    if j.exists():
        return pd.DataFrame(json.loads(j.read_text(encoding="utf-8")))
    csv = (logs_dir / "ultralytics" / model_name / "results.csv").resolve()
    if csv.exists():
        df = pd.read_csv(csv)
        df.columns = [c.strip() for c in df.columns]
        # приводим к нашим именам колонок
        rename = {
            "epoch": "epoch",
            "train/box_loss": "train_loss",
            "metrics/mAP50(B)": "mAP@0.5",
            "metrics/mAP50-95(B)": "mAP@0.5:0.95",
            "metrics/precision(B)": "precision",
            "metrics/recall(B)": "recall",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        if "precision" in df and "recall" in df:
            df["f1"] = 2 * df["precision"] * df["recall"] / (df["precision"] + df["recall"] + 1e-9)
        return df
    return None


def plot_training_curves(model_names: list[str], logs_dir: str, out_dir: str):
    logs_dir, out_dir = Path(logs_dir), ensure_dir(out_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name in model_names:
        df = _read_history(name, logs_dir)
        if df is None or "epoch" not in df:
            continue
        if "train_loss" in df:
            axes[0].plot(df["epoch"], df["train_loss"], label=name)
        if "mAP@0.5" in df:
            axes[1].plot(df["epoch"], df["mAP@0.5"], label=name)
    axes[0].set(title="Loss обучения по эпохам", xlabel="эпоха", ylabel="loss")
    axes[1].set(title="mAP@0.5 на валидации по эпохам", xlabel="эпоха", ylabel="mAP@0.5")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    path = out_dir / "training_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_comparison_bar(summary_csv: str, out_dir: str):
    """Столбчатая диаграмма итоговых метрик по моделям."""
    out_dir = ensure_dir(out_dir)
    df = pd.read_csv(summary_csv)
    metrics = [m for m in ["mAP@0.5", "mAP@0.5:0.95", "f1"] if m in df.columns]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    width = 0.8 / len(metrics)
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, df[m], width, label=m)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(df["model"], rotation=20)
    ax.set(title="Сравнение моделей по итоговым метрикам", ylabel="значение")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "model_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confusion(cm: np.ndarray, class_names: list[str], model_name: str, out_dir: str):
    out_dir = ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(10, 8))
    # нормализуем по строкам (по GT)
    cm_norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    sns.heatmap(
        cm_norm, annot=False, cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=True,
    )
    ax.set(title=f"Матрица ошибок: {model_name}", xlabel="предсказание", ylabel="истинный тип")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    fig.tight_layout()
    path = out_dir / f"confusion_{model_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_metric_vs_param(exp_csv: str, param: str, metric: str, out_dir: str):
    """Зависимость качества от гиперпараметра (раздел 3.7)."""
    out_dir = ensure_dir(out_dir)
    df = pd.read_csv(exp_csv)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, grp in df.groupby("model"):
        grp = grp.sort_values(param)
        ax.plot(grp[param], grp[metric], marker="o", label=name)
    ax.set(title=f"{metric} в зависимости от {param}", xlabel=param, ylabel=metric)
    ax.legend()
    fig.tight_layout()
    path = out_dir / f"{metric}_vs_{param}.png".replace("@", "").replace(":", "")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
