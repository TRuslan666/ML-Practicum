import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt


class FasterRCNNLogger:
    def __init__(self, output_dir="results"):
        self.output_dir = Path(output_dir)

        self.logs_dir = self.output_dir / "logs"
        self.metrics_dir = self.output_dir / "metrics"
        self.plots_dir = self.output_dir / "plots"

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = {
            "epoch": [],
            "train_loss": [],
            "loss_classifier": [],
            "loss_box_reg": [],
            "loss_objectness": [],
            "loss_rpn_box_reg": [],
            "lr": [],
            "skipped_batches": []
        }

    def log_epoch(
        self,
        epoch,
        train_loss,
        loss_classifier,
        loss_box_reg,
        loss_objectness,
        loss_rpn_box_reg,
        lr,
        skipped_batches,
    ):
        self.metrics["epoch"].append(epoch)
        self.metrics["train_loss"].append(train_loss)
        self.metrics["loss_classifier"].append(loss_classifier)
        self.metrics["loss_box_reg"].append(loss_box_reg)
        self.metrics["loss_objectness"].append(loss_objectness)
        self.metrics["loss_rpn_box_reg"].append(loss_rpn_box_reg)
        self.metrics["lr"].append(lr)
        self.metrics["skipped_batches"].append(skipped_batches)

        self.save_json()
        self.save_plots()

    def save_json(self):
        with open(self.metrics_dir / "faster_rcnn_metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=4)

    def save_plots(self):

        epochs = self.metrics["epoch"]

        # -----------------------
        # Общий loss
        # -----------------------
        plt.figure(figsize=(8,5))
        plt.plot(epochs, self.metrics["train_loss"])
        plt.grid(True)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.savefig(self.plots_dir / "train_loss.png")
        plt.close()

        # -----------------------
        # Все компоненты loss
        # -----------------------
        plt.figure(figsize=(10,6))

        plt.plot(epochs, self.metrics["loss_classifier"], label="classifier")
        plt.plot(epochs, self.metrics["loss_box_reg"], label="box_reg")
        plt.plot(epochs, self.metrics["loss_objectness"], label="objectness")
        plt.plot(epochs, self.metrics["loss_rpn_box_reg"], label="rpn_box_reg")

        plt.grid(True)
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Components")

        plt.savefig(self.plots_dir / "loss_components.png")
        plt.close()

        # -----------------------
        # LR
        # -----------------------
        plt.figure(figsize=(8,5))
        plt.plot(epochs, self.metrics["lr"])
        plt.grid(True)
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate")

        plt.savefig(self.plots_dir / "learning_rate.png")
        plt.close()

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