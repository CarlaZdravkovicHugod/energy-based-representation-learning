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

###############################################################################
# Dataset
###############################################################################
class NumpyMRIDataset(Dataset):
    """Loads pre-exported `.npy` magnitude slices.

    Each file can be (H,W) **or** (N,H,W). Channel dim is added automatically.
    Images are min-max scaled to [0,1].
    """

    def __init__(self, root: str | Path, augment: bool = False, size: int | None = None, eval: bool = False):
        self.root = Path(root)
        self.paths = sorted(p for p in self.root.glob("*.npy"))
        self.paths = self.paths[:int(len(self.paths) * 0.8)] if eval else self.paths[int(len(self.paths) * 0.8):]
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

###############################################################################
# Architecture – UNet Lite for Reconstruction
###############################################################################

class SE(nn.Module):                                    # tiny squeeze-excite
    def __init__(self, ch, r=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Conv2d(ch, ch // r, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch // r, ch, 1), nn.Sigmoid())
    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w

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
    r"""
    * skip_connections = True  → classical U-Net
    * skip_connections = False → space-to-depth trick, channel attention,
                                 and NO encoder–decoder skips
    """
    def __init__(self,
                 in_ch: int = 1,
                 out_ch: int = 1,
                 base_ch: int = 32,
                 skip_connections: bool = True,
                 s2d_factor: int = 2):           # ⇐ typically 2 or 4
        super().__init__()
        self.skip_connections = skip_connections
        self.s2d = (not skip_connections)
        self.s2d_factor = s2d_factor if self.s2d else 1
        self.base_ch = base_ch

        # ---------- Encoder --------------------------------------------------
        in_ch_eff = in_ch * self.s2d_factor**2
        self.inc   = DoubleConv(in_ch_eff, base_ch)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch,   base_ch*2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch*2, base_ch*4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch*4, base_ch*8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch*8, base_ch*8))

        # ---------- Decoder --------------------------------------------------
        factor = 2 if skip_connections else 1
        self.up1  = nn.ConvTranspose2d(base_ch*8, base_ch*8, 2, stride=2)
        self.dec1 = nn.Sequential(DoubleConv(base_ch*8*factor, base_ch*4),
                                  SE(base_ch*4))
        self.up2  = nn.ConvTranspose2d(base_ch*4, base_ch*4, 2, stride=2)
        self.dec2 = nn.Sequential(DoubleConv(base_ch*4*factor, base_ch*2),
                                  SE(base_ch*2))
        self.up3  = nn.ConvTranspose2d(base_ch*2, base_ch*2, 2, stride=2)
        self.dec3 = nn.Sequential(DoubleConv(base_ch*2*factor, base_ch),
                                  SE(base_ch))
        self.up4  = nn.ConvTranspose2d(base_ch, base_ch, 2, stride=2)
        self.dec4 = nn.Sequential(DoubleConv(base_ch*factor, base_ch),
                                  SE(base_ch))

        out_mult  = self.s2d_factor**2                 # because we’ll pixel-shuffle
        self.outc = nn.Conv2d(base_ch, out_ch * out_mult, 1)
        self.act  = nn.Sigmoid()

    # ------------------------------------------------------------------------
    def _maybe_s2d(self, x):
        return F.pixel_unshuffle(x, self.s2d_factor) if self.s2d else x

    def _maybe_d2s(self, x):
        return F.pixel_shuffle(x, self.s2d_factor)    if self.s2d else x

    # ------------------------------------------------------------------------
    def encode(self, x):
        x  = self._maybe_s2d(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.skip_connections:
            return x5, (x1, x2, x3, x4)
        return x5, None

    def decode(self, z, skips=None, target_size=None):
        if self.skip_connections and skips:
            x1, x2, x3, x4 = skips
            x = _align(self.up1(z), x4); x = torch.cat([x, x4], 1); x = self.dec1(x)
            x = _align(self.up2(x), x3); x = torch.cat([x, x3], 1); x = self.dec2(x)
            x = _align(self.up3(x), x2); x = torch.cat([x, x2], 1); x = self.dec3(x)
            x = _align(self.up4(x), x1); x = torch.cat([x, x1], 1); x = self.dec4(x)
        else:
            x = self.up1(z); x = self.dec1(x)
            x = self.up2(x); x = self.dec2(x)
            x = self.up3(x); x = self.dec3(x)
            x = self.up4(x); x = self.dec4(x)

        x = self.act(self.outc(x))
        x = self._maybe_d2s(x)
        if target_size and x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear",
                              align_corners=False)
        return x

    def forward(self, x):
        z, skips = self.encode(x)
        return self.decode(z, skips, target_size=x.shape[2:])

    # convenience ------------------------------------------------------------
    def reconstruct(self, x):
        return self.forward(x)

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


def train_epoch(model, loader, optimizer, device, ssim_weight=0.2, epoch=0, neptune_logger=None):
    model.train()
    ssim_loss_fn = get_ssim_loss().to(device)
    total_loss = 0.0
    for i, batch in tqdm(enumerate(loader)):
        batch = batch.to(device)
        latent, skips = model.encode(batch)
        recon = model.decode(latent, skips)
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
        grid = make_grid(torch.cat([batch, recon], 0), nrow=8)
        save_image(grid, f"runs/autoencoder/epoch{epoch},step{i},loss{round(loss.item(), 4)}.png")
        img = Image.open(f"runs/autoencoder/epoch{epoch},step{i},loss{round(loss.item(), 4)}.png")
        neptune_logger.log_image("recons", img, step=epoch * len(loader) + i)
        neptune_logger.log_metric("loss", loss.item(), step=epoch * len(loader) + i)
        os.remove(f"runs/autoencoder/epoch{epoch},step{i},loss{round(loss.item(), 4)}.png")
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, device, epoch=0, neptune_logger=None, ssim_weight=0.2):
    loader_len = len(loader)
    model.eval()
    ssim_loss_fn = get_ssim_loss().to(device)
    total_psnr = 0.0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader)):
            batch = batch.to(device)
            latent, skips = model.encode(batch)
            recon = model.decode(latent, skips)
            _, psnr = mse_psnr(batch, recon)
            total_psnr += psnr * batch.size(0)
            mse_loss = nn.functional.mse_loss(recon, batch)
            if ssim_weight > 0:
                ssim_loss = 1 - ssim_loss_fn(recon, batch).mean()
                loss = (1 - ssim_weight) * mse_loss + ssim_weight * ssim_loss
            else:
                loss = mse_loss
            grid = make_grid(torch.cat([batch, recon], 0), nrow=8)
            save_image(grid, f"runs/autoencoder/eval_epoch{epoch},step{i},loss{round(loss.item(), 4)}.png")
            img = Image.open(f"runs/autoencoder/eval_epoch{epoch},step{i},loss{round(loss.item(), 4)}.png")
            neptune_logger.log_image("eval_recons", img, step=epoch * loader_len + i)
            neptune_logger.log_metric("loss", loss.item(), step=epoch * loader_len + i)
            os.remove(f"runs/autoencoder/eval_epoch{epoch},step{i},loss{round(loss.item(), 4)}.png")
    return loss.item()

###############################################################################
# Main
###############################################################################

def main():
    p = argparse.ArgumentParser(description="Enhanced MRI Autoencoder Trainer")
    p.add_argument("--train_dir", required=True)
    p.add_argument("--description", required=True)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--size", type=int, default=None, help="Resize square length (e.g. 128)")
    p.add_argument("--out", type=str, default="runs/autoencoder")
    args = p.parse_args()
    best_loss = float("inf")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.out).mkdir(parents=True, exist_ok=True)

    train_ds = NumpyMRIDataset(args.train_dir, augment=False, size=args.size)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    val_ds = NumpyMRIDataset(args.train_dir, augment=False, size=args.size, eval=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)

    model = UNetAutoencoder(base_ch=256, skip_connections=False).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    neptune_logger = NeptuneLogger(False, description=args.description)

    for epoch in tqdm(range(1, args.epochs + 1)):
        loss = train_epoch(model, train_loader, optimizer, device, ssim_weight=0.2, epoch=epoch, neptune_logger=neptune_logger)
        eval_loss = eval_epoch(model, val_loader, device, epoch=epoch, neptune_logger=neptune_logger)
        print(f"Epoch {epoch:03d}: train-loss {loss:.4f}, eval-loss {eval_loss:.4f}")

        # Save example grid
        # x = next(iter(train_loader))[:8].to(device)
        # recon = model(x)
        # grid = make_grid(torch.cat([x, recon], 0), nrow=8)
        # save_image(grid, Path(args.out) / f"epoch{epoch:03d}.png")

        # Checkpoint
        if eval_loss < best_loss:
            best_loss = eval_loss
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict()},
                Path(args.out) / "best_model.pt",
            )
            neptune_logger.log_model(Path(args.out) / "best_model.pt", "best_model.pt")

    print("Training complete. Best Loss:", best_loss)


if __name__ == "__main__":
    main()
