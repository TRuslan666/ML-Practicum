from pathlib import Path
from ultralytics import YOLO

# ПАТЧ для .ppm
from ultralytics.data.utils import IMG_FORMATS
IMG_FORMATS.add("ppm")


def train_yolo(
    weights: str,
    data_yaml: str,
    epochs: int,
    imgsz: int,
    batch: int,
    project: str,
    name: str,
    **extra_kwargs,
):
    model = YOLO(weights)

    save_dir = extra_kwargs.pop("save_dir", None)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        save_dir=save_dir,
        **extra_kwargs
    )
    return results


def train_yolov5(config: dict):
    return train_yolo(
        weights=config.get("weights", "yolov5s.pt"),
        data_yaml=config["data_yaml"],
        epochs=config.get("epochs", 50),
        imgsz=config.get("imgsz", 640),
        batch=config.get("batch", 16),
        project=config.get("project", "results"),
        name=config.get("name", "yolov5"),
        save_dir=config.get("save_dir", "results/yolov5"),
        exist_ok=True,
        **{k: v for k, v in config.items() 
           if k not in ["weights", "data_yaml", "epochs", "imgsz", "batch", 
                       "project", "name", "save_dir", "exist_ok"]}
    )


def train_yolov8(config: dict):
    return train_yolo(
        weights=config.get("weights", "yolov8n.pt"),
        data_yaml=config["data_yaml"],
        epochs=config.get("epochs", 50),
        imgsz=config.get("imgsz", 640),
        batch=config.get("batch", 16),
        project=config.get("project", "results"),
        name=config.get("name", "yolov8"),
        save_dir=config.get("save_dir", "results/yolov8"),
        exist_ok=True,
        **{k: v for k, v in config.items() 
           if k not in ["weights", "data_yaml", "epochs", "imgsz", "batch", 
                       "project", "name", "save_dir", "exist_ok"]}
    )