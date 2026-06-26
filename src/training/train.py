from ultralytics import YOLO


def train_yolo(config):
    model = YOLO(config["model_path"])

    results = model.train(
        data=config["dataset"]["path"],
        epochs=config["training"]["epochs"],
        imgsz=config["training"]["imgsz"],
        batch=config["training"]["batch"],
        device=config["training"]["device"],
        amp=config["training"]["amp"],
        cache=config["training"]["cache"],
        workers=config["training"]["workers"]
    )

    return results