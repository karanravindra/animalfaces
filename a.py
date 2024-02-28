# %% [markdown]
# Dataset Source:
# https://www.kaggle.com/datasets/andrewmvd/animal-faces

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.trainer import Trainer
import matplotlib.pyplot as plt
import wandb
import torchmetrics
import numpy as np


# %%
dataset = torchvision.datasets.ImageFolder(
    root='data/afhq/train',
    transform=transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
    ])
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)


# %%
generator = nn.Sequential(
    nn.BatchNorm1d(100),
    nn.Linear(100, 256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Linear(256, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Linear(512, 1024),
    nn.ReLU(),
    nn.BatchNorm1d(1024),
    nn.Linear(1024, 64*64*3),
    nn.Unflatten(1, (3, 64, 64)),
)

discriminator = nn.Sequential(
    nn.Flatten(),
    nn.Linear(64*64*3, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
)

class GAN(L.LightningModule):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.automatic_optimization = False
        
    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()
        real_imgs, real_labels = batch
        
        # Train discriminator
        d_opt.zero_grad()
        z = torch.randn(real_imgs.shape[0], 100, device=self.device)
        fake_imgs = self.generator(z)
        real_preds = self.discriminator(real_imgs)
        fake_preds = self.discriminator(fake_imgs)
        real_loss = F.binary_cross_entropy_with_logits(real_preds, torch.ones_like(real_preds))
        fake_loss = F.binary_cross_entropy_with_logits(fake_preds, torch.zeros_like(fake_preds))
        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_opt.step()
        
        # Train generator
        g_opt.zero_grad()
        z = torch.randn(real_imgs.shape[0], 100, device=self.device)
        fake_imgs = self.generator(z)
        fake_preds = self.discriminator(fake_imgs)
        g_loss = F.binary_cross_entropy_with_logits(fake_preds, torch.ones_like(fake_preds))
        g_loss.backward()
        g_opt.step()


        if batch_idx % 100 == 0:
            z = torch.randn(real_imgs.shape[0], 100, device=self.device)
            fake_imgs = self.generator(z)
            real_preds = self.discriminator(real_imgs)
            fake_preds = self.discriminator(fake_imgs)
            real_loss = F.binary_cross_entropy_with_logits(real_preds, torch.ones_like(real_preds))
            fake_loss = F.binary_cross_entropy_with_logits(fake_preds, torch.zeros_like(fake_preds))
            d_loss = real_loss + fake_loss
            g_loss = F.binary_cross_entropy_with_logits(fake_preds, torch.ones_like(fake_preds))
            
            
            # Calculate FID
            fake_imgs = F.interpolate(fake_imgs, size=(299, 299), mode='nearest').cpu().detach()
            real_imgs = F.interpolate(real_imgs, size=(299, 299), mode='nearest').cpu().detach()

            metric = torchmetrics.image.fid.FrechetInceptionDistance(feature=64, normalize=True)
            
            metric.update(fake_imgs, real=False)
            metric.update(real_imgs, real=True)
            
            fid_score = metric.compute()
            
            self.logger.experiment.log({
                "Generator Loss": g_loss.item(),
                "Discriminator Loss": d_loss.item(),
                "Discriminator Accuracy": (real_preds > 0.5).float().mean().item(),
                "Discriminator Loss Real": real_loss.item(),
                "Discriminator Loss Fake": fake_loss.item(),
                "Generated Images": wandb.Image(torchvision.utils.make_grid(fake_imgs), caption=f"Generated Images Epoch {self.current_epoch}, Batch {batch_idx}"),
                "FID Score": fid_score
                })
        
    def configure_optimizers(self):
        g_opt = optim.Adam(self.generator.parameters(), lr=0.0002)
        d_opt = optim.Adam(self.discriminator.parameters(), lr=0.0001)
        return [g_opt, d_opt], []

gan = GAN(generator, discriminator)


# %%
logger = WandbLogger(project="gan", tags=["fc"])
trainer = Trainer(max_epochs=5, logger=logger, limit_val_batches=1, strategy="ddp_spawn")
trainer.fit(gan, dataloader, dataloader)
wandb.finish()


# %%



