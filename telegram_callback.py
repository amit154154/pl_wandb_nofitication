import os
import requests
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import tempfile
import wandb
from typing import List
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import make_interp_spline

sns.set(style="whitegrid")


class TelegramNotificationCallback(pl.Callback):
    def __init__(self, bot_token=None, chat_id=None, send_image_every_n_steps=1000):
        super().__init__()
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        assert self.bot_token and self.chat_id, "Telegram bot token and chat ID are required"

        self.send_image_every_n_steps = send_image_every_n_steps
        self.last_image_step_sent = -1

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
        # Track loss
        if isinstance(outputs, dict) and "loss" in outputs:
            try:
                self.current_train_step_losses.append(float(outputs["loss"]))
            except:
                pass

        # Send image if available
        global_step = trainer.global_step
        img_path = "last_logged.png"
        if (
            os.path.exists(img_path)
            and global_step % self.send_image_every_n_steps == 0
            and global_step != self.last_image_step_sent
        ):
            self.last_image_step_sent = global_step
            try:
                self._send_telegram_message(f"📸 Sample Image @ Step {global_step}", image_path=img_path)
                # Optional cleanup:
                # os.remove(img_path)
            except Exception as e:
                print(f"[TelegramCallback] Image send error: {e}")

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics

        # Store losses
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

        # Generate plots
        image_paths = []

        # 1. Train loss per epoch
        if self.train_epoch_losses:
            image_paths.append(self._make_plot(
                self.train_epoch_losses,
                title="Train Loss per Epoch",
                ylabel="Loss",
                xlabel="Epoch"
            ))

        # 2. Train loss during current epoch (smoothed)
        if self.current_train_step_losses:
            smoothed = self._smooth_curve(self.current_train_step_losses, window=20)
            image_paths.append(self._make_plot(
                smoothed,
                title="Train Loss during Current Epoch",
                ylabel="Loss",
                xlabel="Batch Step",
                smooth=False
            ))
            self.current_train_step_losses.clear()

        # 3. Validation loss per epoch
        if self.val_epoch_losses:
            image_paths.append(self._make_plot(
                self.val_epoch_losses,
                title="Validation Loss per Epoch",
                ylabel="Val Loss",
                xlabel="Epoch"
            ))

        # Send message + plots
        self._send_telegram_message(message)
        for path in image_paths:
            self._send_telegram_message(image_path=path)

    def on_train_end(self, trainer, pl_module):
        self._send_telegram_message(f"✅ Training complete for `{pl_module.__class__.__name__}`\n🔗 [W&B Run]({self.run_url})")

    def _smooth_curve(self, values, window=20):
        values = np.array(values)
        if len(values) < window:
            return values
        return np.convolve(values, np.ones(window) / window, mode='valid')

    def _make_plot(self, data, title, ylabel, xlabel, smooth=False):
        fig, ax = plt.subplots(figsize=(6, 3.5), dpi=200)

        x = np.arange(1, len(data) + 1)
        y = np.array(data)

        if smooth and len(x) > 3:
            x_smooth = np.linspace(x.min(), x.max(), 200)
            spline = make_interp_spline(x, y, k=3)
            y_smooth = spline(x_smooth)
        else:
            x_smooth = x
            y_smooth = y

        cmap = LinearSegmentedColormap.from_list("gradient", ["#36D1DC", "#5B86E5"])
        for i in range(len(x_smooth) - 1):
            ax.plot(x_smooth[i:i + 2], y_smooth[i:i + 2], color=cmap(i / len(x_smooth)), linewidth=2.5,
                    solid_capstyle='round')

        if not smooth:
            ax.scatter(x, y, s=35, color="#36D1DC", edgecolors="white", linewidth=1, zorder=3)

        ax.fill_between(x_smooth, y_smooth, y.min() - 0.05 * abs(y.min()), color="#36D1DC", alpha=0.1)

        ax.set_title(title, fontsize=14, fontweight='bold', pad=12, color="#333333")
        ax.set_xlabel(xlabel, fontsize=11, color="#555555")
        ax.set_ylabel(ylabel, fontsize=11, color="#555555")
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.7)

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