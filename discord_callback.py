# discord_callback.py
import os
import requests
import tempfile
import wandb
import pytorch_lightning as pl

class DiscordNotificationCallback(pl.Callback):
    def __init__(self, webhook_url=None):
        super().__init__()
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
        assert self.webhook_url is not None, "Missing Discord webhook URL"
        self.run_url = None
        self.last_img_id = None

    def _send_discord_message(self, content, image_path=None):
        data = {"content": content}
        files = {"file": open(image_path, "rb")} if image_path else None
        response = requests.post(self.webhook_url, data=data, files=files)
        if response.status_code not in [200, 204]:
            print("❌ Discord failed:", response.text)

    def on_train_start(self, trainer, pl_module):
        if wandb.run:
            self.run_url = wandb.run.get_url()
        self._send_discord_message(f"🚀 Training started for `{pl_module.__class__.__name__}`.\n🔗 [W&B Run]({self.run_url})")

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        print(f"[DEBUG] Epoch {epoch + 1} ended — metrics: {metrics}")  # Confirm callback fires

        # Format all float/int metrics
        metric_lines = [
            f"- **{k}**: `{float(v):.4f}`"
            for k, v in metrics.items() if isinstance(v, (float, int))
        ]
        message = f"📊 **Epoch {epoch + 1}** Complete\n" + "\n".join(metric_lines)

        image_path = "last_logged.png" if os.path.exists("last_logged.png") else None
        self._send_discord_message(message, image_path=image_path)

    def on_train_end(self, trainer, pl_module):
        self._send_discord_message(f"✅ Training complete for `{pl_module.__class__.__name__}`.\n🔗 [W&B Run]({self.run_url})")