import logging
from pathlib import Path


def setup_logger():
    Path("results/logs").mkdir(
        parents=True,
        exist_ok=True
    )

    logging.basicConfig(
        filename="results/logs/train.log",
        level=logging.INFO,
        format="%(asctime)s - %(message)s"
    )

    return logging