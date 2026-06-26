from ultralytics.data import utils

from src.training.train_yolov8 import train_yolo
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.evaluation.save_metrics import save_metrics
from src.evaluation.save_metrics import save_plots


utils.IMG_FORMATS.add("ppm")


def main():

    logger = setup_logger()

    logger.info("Loading config")

    config = load_config(
        "configs/yolov8.yaml"
    )

    logger.info("Training started")

    train_yolo(config)

    save_metrics()
    
    save_plots()

    logger.info("Training finished")


if __name__ == "__main__":
    main()