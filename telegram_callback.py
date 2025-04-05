import os
import requests
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import tempfile
import wandb
from typing import List
import seaborn as sns
sns.set(style="whitegrid")  # Apply nice background style
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import make_interp_spline


class TelegramNotificationCallback(pl.Callback):
    def __init__(self, bot_token=None, chat_id=None):
        super().__init__()
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        assert self.bot_token and self.chat_id, "Telegram bot token and chat ID are required"

        self.run_url = None
        self.train_epoch_losses: List[float] = []
        self.val_epoch_losses: List[float] = []
        self.current_train_step_losses: List[float] = []

    def _send_telegram_message(self, text=None, image_path=None):
        base_url = f"https://api.telegram.org/bot{self.bot_token}"
        if text:
            requests.post(f"{base_url}/sendMessage", data={"chat_id": self.chat_id, "text": text})
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as img:
                requests.post(f"{base_url}/sendPhoto", data={"chat_id": self.chat_id}, files={"photo": img})

    def on_train_start(self, trainer, pl_module):
        if wandb.run:
            self.run_url = wandb.run.get_url()
        self._send_telegram_message(
            f"🚀 Training started for `{pl_module.__class__.__name__}`\n🔗 [W&B Run]({self.run_url})"
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if isinstance(outputs, dict) and "loss" in outputs:
            try:
                self.current_train_step_losses.append(float(outputs["loss"]))
            except:
                pass

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics

        # Extract losses
        train_epoch_loss = metrics.get("train_loss_epoch") or metrics.get("train_loss")
        val_loss = metrics.get("val_loss")

        try:
            if train_epoch_loss is not None:
                self.train_epoch_losses.append(float(train_epoch_loss))
        except:
            pass

        try:
            if val_loss is not None:
                self.val_epoch_losses.append(float(val_loss))
        except:
            pass

        # Format metrics message
        lines = []
        for k, v in metrics.items():
            try:
                lines.append(f"- {k}: {float(v):.4f}")
            except (TypeError, ValueError):
                continue
        message = f"📊 Epoch {epoch+1} Summary\n" + "\n".join(lines)

        # Generate 3 plots
        image_paths = []

        # 1. train loss per epoch
        if self.train_epoch_losses:
            image_paths.append(self._make_plot(
                self.train_epoch_losses,
                title="Train Loss per Epoch",
                ylabel="Loss",
                xlabel="Epoch"
            ))

        # 2. train loss during current epoch
        if self.current_train_step_losses:
            smoothed = self._smooth_curve(self.current_train_step_losses, window=20)
            image_paths.append(self._make_plot(
                smoothed,
                title="Train Loss during Current Epoch",
                ylabel="Loss",
                xlabel="Batch Step",
                smooth=False  # Already smoothed data
            ))
            self.current_train_step_losses.clear()

        # 3. validation loss per epoch
        if self.val_epoch_losses:
            image_paths.append(self._make_plot(
                self.val_epoch_losses,
                title="Validation Loss per Epoch",
                ylabel="Val Loss",
                xlabel="Epoch"
            ))

        # Send text + each plot
        self._send_telegram_message(message)
        for img_path in image_paths:
            self._send_telegram_message(image_path=img_path)

    def on_train_end(self, trainer, pl_module):
        self._send_telegram_message(f"✅ Training complete for `{pl_module.__class__.__name__}`\n🔗 [W&B Run]({self.run_url})")

    def _smooth_curve(self, values, window=20):
        import numpy as np
        values = np.array(values)
        if len(values) < window:
            return values  # not enough points
        return np.convolve(values, np.ones(window) / window, mode='valid')
    def _make_plot(self, data, title, ylabel, xlabel, smooth=False):
        fig, ax = plt.subplots(figsize=(6, 3.5), dpi=200)

        x = np.arange(1, len(data) + 1)
        y = np.array(data)

        # Apply smoothing if requested and enough points exist
        if smooth and len(x) > 3:
            x_smooth = np.linspace(x.min(), x.max(), 200)
            spline = make_interp_spline(x, y, k=3)
            y_smooth = spline(x_smooth)
        else:
            x_smooth = x
            y_smooth = y

        # Gradient style
        cmap = LinearSegmentedColormap.from_list("gradient", ["#36D1DC", "#5B86E5"])
        for i in range(len(x_smooth) - 1):
            ax.plot(x_smooth[i:i + 2], y_smooth[i:i + 2], color=cmap(i / len(x_smooth)), linewidth=2.5,
                    solid_capstyle='round')

        # Scatter dots only if not smoothed
        if not smooth:
            ax.scatter(x, y, s=35, color="#36D1DC", edgecolors="white", linewidth=1, zorder=3)

        # Shadow under curve
        ax.fill_between(x_smooth, y_smooth, y.min() - 0.05 * abs(y.min()), color="#36D1DC", alpha=0.1)

        # Labels & formatting
        ax.set_title(title, fontsize=14, fontweight='bold', pad=12, color="#333333")
        ax.set_xlabel(xlabel, fontsize=11, color="#555555")
        ax.set_ylabel(ylabel, fontsize=11, color="#555555")
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.7)

        # Style cleanup
        sns.despine(left=True, bottom=True)
        fig.patch.set_facecolor("#F7F9FC")
        ax.set_facecolor("#F7F9FC")
        for spine in ax.spines.values():
            spine.set_visible(False)

        fig.tight_layout()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(tmp.name, bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)
        return tmp.name