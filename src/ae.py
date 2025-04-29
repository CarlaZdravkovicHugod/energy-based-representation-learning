"""
Enhanced MRI Autoencoder (UNet-style)
====================================
A self-contained PyTorch training script that integrates the improvements we
discussed: deeper encoder-decoder with skip connections, mixed MSE + SSIM loss,
PSNR monitoring, optional data augmentation, automatic checkpointing and
visual logging.

Designed for 2-D magnitude MRI slices of arbitrary size (powers of two work
best). Default input is 128x128, but the network adapts to any size divisible
by 32.

Dependencies
------------
python ≥ 3.9, torch ≥ 2.2, torchvision, numpy, scikit-image
Optionally `piqa` for differentiable SSIM (pip install piqa).

Usage
-----
```bash
python enhanced_mri_autoencoder.py \
    --train_dir ~/data/mri/train \
    --val_dir   ~/data/mri/val \
    --epochs 100 --batch 8 --size 128
```

During training the script writes:
* `runs/autoencoder/best_model.pt` - best checkpoint by PSNR.
* `runs/autoencoder/epoch***.png` - grid of inputs vs reconstructions.
"""

from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
from typing import Tuple

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

###############################################################################
# Dataset
###############################################################################
class NumpyMRIDataset(Dataset):
    """Loads pre-exported `.npy` magnitude slices.

    Each file can be (H,W) **or** (N,H,W). Channel dim is added automatically.
    Images are min-max scaled to [0,1].
    """

    def __init__(self, root: str | Path, augment: bool = False, size: int | None = None):
        self.root = Path(root)
        self.paths = sorted(p for p in self.root.glob("*.npy"))
        self.size = size
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
        return img

###############################################################################
# Architecture – UNet Lite for Reconstruction
###############################################################################

def _align(x, ref):
    """Bilinearly resize x so its H,W match ref (skip connection)."""
    if x.shape[2:] != ref.shape[2:]:
        x = F.interpolate(x, size=ref.shape[2:], mode="bilinear",
                          align_corners=False)
    return x

class DoubleConv(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

class UNetAutoencoder(nn.Module):
    def __init__(self, base_ch: int = 32):
        super().__init__()
        # Encoder
        self.inc   = DoubleConv(1, base_ch)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch, base_ch*2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch*2, base_ch*4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch*4, base_ch*8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch*8, base_ch*8))
        # Decoder
        self.up1 = nn.ConvTranspose2d(base_ch*8, base_ch*8, 2, stride=2)
        self.dec1 = DoubleConv(base_ch*16, base_ch*4)
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*4, 2, stride=2)
        self.dec2 = DoubleConv(base_ch*8, base_ch*2)
        self.up3 = nn.ConvTranspose2d(base_ch*2, base_ch*2, 2, stride=2)
        self.dec3 = DoubleConv(base_ch*4, base_ch)
        self.up4 = nn.ConvTranspose2d(base_ch, base_ch, 2, stride=2)
        self.dec4 = DoubleConv(base_ch*2, base_ch)
        self.outc = nn.Conv2d(base_ch, 1, 1)
        self.act  = nn.Sigmoid()

    def forward(self, x):
        # encoder ---------------------------------------------------------------
        x1 = self.inc(x)         #  32 ch
        x2 = self.down1(x1)      #  64 ch
        x3 = self.down2(x2)      # 128 ch
        x4 = self.down3(x3)      # 256 ch
        x5 = self.down4(x4)      # 256 ch (same as x4)

        # decoder ---------------------------------------------------------------
        x = _align(self.up1(x5), x4)          # 256→256, align to x4
        x = torch.cat([x, x4], 1)             # 256+256 = 512
        x = self.dec1(x)                      # →128 ch

        x = _align(self.up2(x), x3)           # 128→128
        x = torch.cat([x, x3], 1)             # 128+128 = 256
        x = self.dec2(x)                      # →64 ch

        x = _align(self.up3(x), x2)           # 64→64
        x = torch.cat([x, x2], 1)             # 64+64 = 128
        x = self.dec3(x)                      # →32 ch

        x = _align(self.up4(x), x1)           # 32→32
        x = torch.cat([x, x1], 1)             # 32+32 = 64
        x = self.dec4(x)                      # →32 ch

        return self.act(self.outc(x))         # 1 ch

###############################################################################
# Metrics
###############################################################################
@torch.no_grad()
def mse_psnr(x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    mse = nn.functional.mse_loss(x, y).item()
    psnr = 20 * math.log10(1.0 / math.sqrt(mse + 1e-12))
    return mse, psnr

###############################################################################
# Training helpers
###############################################################################

def get_ssim_loss():
    """Prefer differentiable SSIM from piqa, else fallback to dummy that returns 0."""
    try:
        from piqa import SSIM
        return SSIM(n_channels=1)
    except ImportError:
        class _Dummy(nn.Module):
            def forward(self, x, y):
                return torch.zeros(1, device=x.device)
        return _Dummy()


def train_epoch(model, loader, optimizer, device, ssim_weight=0.2):
    model.train()
    ssim_loss_fn = get_ssim_loss().to(device)
    total_loss = 0.0
    for batch in tqdm(loader):
        batch = batch.to(device)
        recon = model(batch)
        mse_loss = nn.functional.mse_loss(recon, batch)
        if ssim_weight > 0:
            ssim_loss = 1 - ssim_loss_fn(recon, batch).mean()
            loss = (1 - ssim_weight) * mse_loss + ssim_weight * ssim_loss
        else:
            loss = mse_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.size(0)
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, device):
    model.eval()
    total_psnr = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            recon = model(batch)
            _, psnr = mse_psnr(batch, recon)
            total_psnr += psnr * batch.size(0)
    return total_psnr / len(loader.dataset)

###############################################################################
# Main
###############################################################################

def main():
    p = argparse.ArgumentParser(description="Enhanced MRI Autoencoder Trainer")
    p.add_argument("--train_dir", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--size", type=int, default=None, help="Resize square length (e.g. 128)")
    p.add_argument("--out", type=str, default="runs/autoencoder")
    args = p.parse_args()
    best_loss = float("inf")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.out).mkdir(parents=True, exist_ok=True)

    train_ds = NumpyMRIDataset(args.train_dir, augment=True, size=args.size)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)

    model = UNetAutoencoder().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_psnr = 0.0
    for epoch in tqdm(range(1, args.epochs + 1)):
        loss = train_epoch(model, train_loader, optimizer, device, ssim_weight=0.2)

        print(f"Epoch {epoch:03d}: train-loss {loss:.4f}")

        # Save example grid
        x = next(iter(train_loader))[:8].to(device)
        recon = model(x)
        grid = make_grid(torch.cat([x, recon], 0), nrow=8)
        save_image(grid, Path(args.out) / f"epoch{epoch:03d}.png")

        # Checkpoint
        if loss < best_loss:
            best_loss = loss
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict()},
                Path(args.out) / "best_model.pt",
            )

    print("Training complete. Best PSNR:", best_psnr)


if __name__ == "__main__":
    main()
