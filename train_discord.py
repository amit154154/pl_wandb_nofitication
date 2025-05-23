# train.py

import os
import torch
import wandb
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from discord_callback import DiscordNotificationCallback
from torchvision.utils import make_grid
import torchvision
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torchvision
import wandb

# Set your Discord webhook (secure version: set in terminal instead)
os.environ["DISCORD_WEBHOOK_URL"] = "https://discord.com/api/webhooks/your_webhook_url"

# Start W&B
wandb_logger = WandbLogger(project="mnist-discord-demo")

# LightningModule
class LitMNIST(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        return self.layer_2(torch.relu(self.layer_1(x.view(x.size(0), -1))))

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)

        if batch_idx == 0:
            grid = make_grid(x[:8])
            npimg = grid.permute(1, 2, 0).cpu().numpy()

            plt.imsave("last_logged.png", npimg)
            wandb.log({"sample_images": wandb.Image("last_logged.png")})

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# Data
transform = transforms.Compose([transforms.ToTensor()])
mnist_full = MNIST(root=".", train=True, download=True, transform=transform)
mnist_train, mnist_val = random_split(mnist_full, [55000, 5000])
train_loader = DataLoader(mnist_train, batch_size=64)
val_loader = DataLoader(mnist_val, batch_size=64)

# Train
model = LitMNIST()
trainer = Trainer(
    max_epochs=10,
    logger=wandb_logger,
    callbacks=[DiscordNotificationCallback()],
)
trainer.fit(model, train_loader, val_loader)