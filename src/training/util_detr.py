import os
import matplotlib.pyplot as plt
from transformers import TrainerCallback

class PlotMetricsCallback(TrainerCallback):
    def __init__(self, output_dir="results/plots"):
        self.output_dir = output_dir

    def on_train_end(self, args, state, control, **kwargs):
        """Этот метод автоматически вызывается, когда trainer.train() закончен"""
        print("\n[Callback] Обучение завершено. Рисуем графики метрик...")
        os.makedirs(self.output_dir, exist_ok=True)
        
        history = state.log_history
        train_logs = [log for log in history if "loss" in log and "epoch" in log]
        
        if not train_logs:
            return

        epochs = [log["epoch"] for log in train_logs]
        losses = [float(log["loss"]) for log in train_logs]
        
        # Строим график потерь
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, losses, 'b-', linewidth=2, label='Train Loss')
        
        # Добавляем валидацию, если она есть
        eval_logs = [log for log in history if "eval_loss" in log and "epoch" in log]
        if eval_logs:
            plt.plot([l["epoch"] for l in eval_logs], [float(l["eval_loss"]) for l in eval_logs], 'r--', label='Val Loss')
            
        plt.title('DETR Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        # Сохраняем в папку по требованиям файла ТЗ
        plt.savefig(os.path.join(self.output_dir, 'detr_loss.png'), dpi=300)
        plt.close()
        print(f"[Callback] График сохранен в {self.output_dir}/detr_loss.png")