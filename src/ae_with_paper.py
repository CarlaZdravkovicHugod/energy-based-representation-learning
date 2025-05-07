
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
from typing import Tuple
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
from tqdm import tqdm
from src.utils.neptune_logger import NeptuneLogger
import torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from utils.neptune_logger import NeptuneLogger
from torch.optim.lr_scheduler import CosineAnnealingLR


class NumpyMRIDataset(Dataset):
    """Loads pre-exported `.npy` magnitude slices.

    Each file can be (H,W) **or** (N,H,W). Channel dim is added automatically.
    Images are min-max scaled to [0,1].
    """

    def __init__(self, root: str | Path, augment: bool = False, size: int | None = None, eval: bool = False):
        self.root = Path(root)
        self.paths = sorted(p for p in self.root.glob("*.npy"))
        self.paths = self.paths[:int(len(self.paths) * 0.8)] if not eval else self.paths[int(len(self.paths) * 0.8):]
        self.size = size
        self.augment = augment
        aug_list = [T.RandomHorizontalFlip(), T.RandomVerticalFlip()] if augment else []
        if size:
            aug_list.append(T.Resize((size, size), antialias=True))
        self.transform = T.Compose(aug_list)

        self.slices: list[np.ndarray] = []
        for path in self.paths:
            arr = np.load(path)
            if arr.ndim == 2:
                arr = arr[None, ...]  # (1,H,W)
            self.slices.extend(arr)  # list of (H,W)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        img = self.slices[idx].astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        img = torch.from_numpy(img)[None, ...]  # (1,H,W)
        img = self.transform(img)
        img = img.unsqueeze(0)
        img = F.interpolate(img, size=(256, 256), mode="bilinear", align_corners=False)
        img = img.squeeze(0)
        return img
    
class MaskedAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 128x128
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 64x64 
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 16x16
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # 64x64
            nn.ReLU(), 
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), # 256x256
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
### ----- ###
### UTILS ###
### ----- ###

def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    # get random batch
    batch = next(iter(loader))
    batch = batch.to(device)
    recon = model(batch)
    loss = ae_loss(batch, recon)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item(), recon, batch

def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    for i, batch in tqdm(enumerate(loader)):
        batch = batch.to(device)
        recon = model(batch)
        loss = ae_loss(batch, recon)
        total_loss += loss.item() * batch.size(0)
    return total_loss / len(loader.dataset), recon, batch

def ae_loss(x, recon):
    return nn.functional.mse_loss(recon, x)

def save_grids(train_batch, train_recon, eval_batch, eval_recon, neptune_logger, epoch):
    grid = make_grid(torch.cat([train_batch, train_recon], 0), nrow=8)
    save_image(grid, "runs/autoencoder/train_recon.png")
    img = Image.open("runs/autoencoder/train_recon.png")
    neptune_logger.log_image("train_recon", img, step = epoch)

    grid = make_grid(torch.cat([eval_batch, eval_recon], 0), nrow=8)
    save_image(grid, "runs/autoencoder/eval_recon.png")
    img = Image.open("runs/autoencoder/eval_recon.png")
    neptune_logger.log_image("eval_recon", img, step = epoch)

    os.remove("runs/autoencoder/train_recon.png")
    os.remove("runs/autoencoder/eval_recon.png")

    
def main():
    p = argparse.ArgumentParser(description="Enhanced MRI Autoencoder Trainer")
    p.add_argument("--train_dir", required=True)
    p.add_argument("--description", required=True)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--size", type=int, default=None, help="Resize square length (e.g. 128)")
    p.add_argument("--out", type=str, default="runs/autoencoder")
    p.add_argument("--test", type=bool, default=False)
    args = p.parse_args()
    best_loss = float("inf")

    neptune_logger = NeptuneLogger(args.test, args.description)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_ds = NumpyMRIDataset(args.train_dir, eval=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)

    val_ds = NumpyMRIDataset(args.train_dir, eval=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    model = MaskedAutoencoder().to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, args.steps)

    for step in tqdm(range(args.steps)):
        train_loss, train_recon, train_batch = train_epoch(model, train_loader, opt, sched, device)
        neptune_logger.log_metric("train_loss", train_loss)
        neptune_logger.log_metric("lr", opt.param_groups[0]["lr"])
        neptune_logger.log_metric("step", step)

        if step % 50 == 0:
            eval_loss, eval_recon, eval_batch = eval_epoch(model, val_loader, device)
            neptune_logger.log_metric("eval_loss", eval_loss)
            save_grids(train_batch, train_recon, eval_batch, eval_recon, neptune_logger, step)
            if eval_loss < best_loss:
                best_loss = eval_loss
                torch.save(
                    {"step": step, "model_state": model.state_dict()},
                    str(Path(args.out) / "best_model.pt"),
                )
                neptune_logger.log_model(str(Path(args.out) / "best_model.pt"), "best_model.pt")
        
if __name__ == "__main__":
    main()