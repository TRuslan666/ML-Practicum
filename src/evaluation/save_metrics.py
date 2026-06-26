import shutil
from pathlib import Path


def save_metrics():

    Path("results/metrics").mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy(
        "runs/detect/train/results.csv",
        "results/metrics/results.csv"
    )

def save_plots():

    Path("results/plots").mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy(
        "runs/detect/train/results.png",
        "results/plots/results.png"
    )