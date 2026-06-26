import shutil
from pathlib import Path


def save_plots():

    source = Path(
        "runs/detect/train/results.png"
    )

    target_dir = Path(
        "results/plots"
    )

    target_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if source.exists():
        shutil.copy(
            source,
            target_dir / "results.png"
        )