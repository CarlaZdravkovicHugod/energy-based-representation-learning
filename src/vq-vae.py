from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
from typing import Tuple
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr

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
from torch.distributions import Normal, kl_divergence
from utils.neptune_logger import NeptuneLogger
from torch.optim.lr_scheduler import CosineAnnealingLR
from piqa import MS_SSIM
from compressai.layers import GDN

# ---------------------------------------------------------------------------- #
# Metric (no gradients)
# ---------------------------------------------------------------------------- #
ms_ssim = MS_SSIM(n_channels=1).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


# =============================================================================
# DATASET
# =============================================================================
class NumpyMRIDataset(Dataset):
    """Loads pre-exported .npy magnitude slices.

    Each file can be (H,W) **or** (N,H,W). Channel dim is added automatically.
    Images are min-max scaled to [0,1].
    """

    def __init__(self, root: str | Path, augment: bool = False, size: int | None = None, eval: bool = False):
        self.root = Path(root)
        self.paths = sorted(p for p in self.root.glob("*.npy"))
        self.paths = self.paths[: int(len(self.paths) * 0.8)] if not eval else self.paths[int(len(self.paths) * 0.8) :]
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


# =============================================================================
# VQ‑VAE COMPONENTS
# =============================================================================
class VectorQuantizer(nn.Module):
    """Basic VQ layer from \"Neural Discrete Representation Learning\" (Oord et al. 2017)."""

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:
            z_e: (B,C,H,W) continuous encoder output
        Returns:
            z_q: quantised tensor, same shape as z_e
            loss: codebook + commitment loss term
        """
        # Flatten input: (B,C,H,W) -> (B*H*W, C)
        flat = z_e.permute(0, 2, 3, 1).contiguous()
        flat = flat.view(-1, self.embedding_dim)

        # Compute distances (||x - e||^2 = ||x||^2 + ||e||^2 - 2 x·e)
        # (N, 1)
        x2 = (flat ** 2).sum(dim=1, keepdim=True)
        e2 = (self.embedding.weight ** 2).sum(dim=1)
        # (N, K)
        distances = x2 + e2 - 2 * torch.matmul(flat, self.embedding.weight.t())

        # Map to closest embedding index
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat.dtype)

        # Quantise and reshape back
        quantised = torch.matmul(encodings, self.embedding.weight).view(*z_e.permute(0, 2, 3, 1).shape)
        z_q = quantised.permute(0, 3, 1, 2).contiguous()

        # Losses
        e_latent_loss = F.mse_loss(z_q.detach(), z_e)
        q_latent_loss = F.mse_loss(z_q, z_e.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight‑through estimator
        z_q = z_e + (z_q - z_e).detach()

        return z_q, loss


# =============================================================================
# VQ‑VAE NETWORK
# =============================================================================
class VQVAE(nn.Module):
    def __init__(self, num_embeddings: int = 512, embedding_dim: int = 256, commitment_cost: float = 0.25):
        super().__init__()

        # ------------------------------------------------------------------ #
        # Encoder = Conv → GDN  (no ReLU). First block uses GroupNorm to
        # stabilise statistics before the divisive normalisation kicks in.
        # ------------------------------------------------------------------ #
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 256 → 128
            nn.GroupNorm(8, 32),  # (Wu & He 2018)
            GDN(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 128 → 64
            GDN(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 64 → 32
            GDN(128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 32 → 16
            GDN(256),
            nn.Conv2d(256, embedding_dim, 3, stride=2, padding=1),  # 16 → 8 (C = embedding_dim)
            GDN(embedding_dim),
            nn.Conv2d(embedding_dim, embedding_dim, 3, stride=2, padding=1),  # 8 → 4
            GDN(embedding_dim),
        )

        # VQ layer
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

        # ------------------------------------------------------------------ #
        # Decoder = IGDN → Deconv  (mirror of encoder)
        # ------------------------------------------------------------------ #
        self.decoder_conv = nn.Sequential(
            GDN(embedding_dim, inverse=True),
            nn.ConvTranspose2d(embedding_dim, embedding_dim, 3, stride=2, padding=1, output_padding=1),  # 4 → 8
            GDN(embedding_dim, inverse=True),
            nn.ConvTranspose2d(embedding_dim, 256, 3, stride=2, padding=1, output_padding=1),  # 8 → 16
            GDN(256, inverse=True),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),  # 16 → 32
            GDN(128, inverse=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 32 → 64
            GDN(64, inverse=True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 64 → 128
            GDN(32, inverse=True),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),  # 128 → 256
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------ #
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z_e = self.encoder_conv(x)

        return z_e

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decoder_conv(z_q)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_e = self.encode(x)
        z_q, vq_loss = self.vq(z_e)
        x_hat = self.decode(z_q)
        return x_hat, vq_loss


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================

def vqvae_loss(x: torch.Tensor, recon: torch.Tensor, vq_loss: torch.Tensor) -> torch.Tensor:
    """Overall loss = reconstructed similarity + codebook loss"""
    rec_loss = 0.8 * F.mse_loss(recon, x) + 0.2 * (1 - ms_ssim(recon, x))
    return rec_loss + vq_loss  # weighting of vq_loss is inside VectorQuantizer


def train_epoch(model: VQVAE, loader: DataLoader, optimizer: optim.Optimizer, scheduler, device):
    model.train()
    batch = next(iter(loader))
    batch = batch.to(device)
    recon, vq_loss = model(batch)
    loss = vqvae_loss(batch, recon, vq_loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item(), recon, batch


@torch.no_grad()
def eval_epoch(model: VQVAE, loader: DataLoader, device):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        recon, vq_loss = model(batch)
        loss = vqvae_loss(batch, recon, vq_loss)
        total_loss += loss.item() * batch.size(0)
    return total_loss / len(loader.dataset), recon, batch


# =============================================================================
# VISUALISATION
# =============================================================================

def save_grids(train_batch, train_recon, eval_batch, eval_recon, neptune_logger, epoch):
    grid = make_grid(torch.cat([train_batch, train_recon], 0), nrow=8)
    save_image(grid, "runs/vqvae/train_recon.png")
    img = Image.open("runs/vqvae/train_recon.png")
    neptune_logger.log_image("train_recon", img, step=epoch)

    grid = make_grid(torch.cat([eval_batch, eval_recon], 0), nrow=8)
    save_image(grid, "runs/vqvae/eval_recon.png")
    img = Image.open("runs/vqvae/eval_recon.png")
    neptune_logger.log_image("eval_recon", img, step=epoch)

    os.remove("runs/vqvae/train_recon.png")
    os.remove("runs/vqvae/eval_recon.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="MRI VQ‑VAE Trainer")
    p.add_argument("--train_dir", required=True)
    p.add_argument("--description", required=True)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--size", type=int, default=None, help="Resize square length (e.g. 128)")
    p.add_argument("--out", type=str, default="runs/vqvae")
    p.add_argument("--test", type=bool, default=False)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    best_loss = float("inf")

    neptune_logger = NeptuneLogger(args.test, args.description)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = NumpyMRIDataset(args.train_dir, eval=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)

    val_ds = NumpyMRIDataset(args.train_dir, eval=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    model = VQVAE(num_embeddings=512, embedding_dim=256, commitment_cost=0.25).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, args.steps)

    for step in tqdm(range(args.steps)):
        train_loss, train_recon, train_batch = train_epoch(model, train_loader, opt, sched, device)
        neptune_logger.log_metric("train_loss", train_loss)
        neptune_logger.log_metric("lr", opt.param_groups[0]["lr"])
        neptune_logger.log_metric("step", step)

        if step % 50 == 0:
            eval_loss, eval_recon, eval_batch = eval_epoch(model, val_loader, device)
            with torch.no_grad():
                batch_np = eval_batch.squeeze(1).cpu().numpy()  # -> (B, H, W)
                recon_np = eval_recon.squeeze(1).cpu().numpy()

                per_ssim = [ssim(b, r, data_range=1.0) for b, r in zip(batch_np, recon_np)]
                per_psnr = [psnr(b, r, data_range=1.0) for b, r in zip(batch_np, recon_np)]

                val_ssim = np.mean(per_ssim)
                val_psnr = np.mean(per_psnr)

            neptune_logger.log_metric("eval_ssim", val_ssim)
            neptune_logger.log_metric("eval_psnr", val_psnr)
            neptune_logger.log_metric("eval_loss", eval_loss)
            save_grids(train_batch, train_recon, eval_batch, eval_recon, neptune_logger, step)

            if eval_loss < best_loss:
                best_loss = eval_loss
                ckpt_path = Path(args.out) / "best_model.pt"
                torch.save({"step": step, "model_state": model.state_dict()}, str(ckpt_path))
                neptune_logger.log_model(str(ckpt_path), "best_model.pt")


if __name__ == "__main__":
    main()
