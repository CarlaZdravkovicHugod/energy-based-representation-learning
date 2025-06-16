
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
from typing import Tuple
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
import logging


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
from comet_models import LatentEBM, LatentEBM128
from config.load_config import load_config
from src.ae_with_paper import MaskedAutoencoder

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
        # img = F.interpolate(img, size=(256, 256), mode="bilinear", align_corners=False)
        img = img.squeeze(0)
        return img
    
class DummyDataset:
    def __init__(self, len_dataset):
        self.len_dataset = len_dataset

    def __len__(self):
        return self.len_dataset
    

def load_ebm_model(len_dataset):
    config_path = "src/config/2DMRI_config.yml"
    model_type = "LatentEBM128"
    checkpoint_path = "/zhome/75/a/187019/dev/energy-based-representation-learning/src/models_10100.pth.pth"
    """
    Test a model on a given dataset and visualize the results.

    Args:
        config_path (str): Path to the configuration file.
        dataset_type (str): Type of dataset ('MRI2D', 'Clevr', etc.).
        model_type (str): Type of model ('LatentEBM', 'LatentEBM128').
        checkpoint_path (str): Path to the model checkpoint.
        num_steps (int): Number of steps for image generation.
        batch_size (int): Batch size for testing.
    """
    # Load configuration
    config = load_config(config_path)
    logging.info(f"Loaded config: {config}")

    run_name = checkpoint_path.split("/")[-1].split(".")[0]
    logging.info(f"Run name: {run_name}")

    dataset = DummyDataset(len_dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if model_type == "LatentEBM":
        model_class = LatentEBM
    elif model_type == "LatentEBM128":
        model_class = LatentEBM128
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dicts = checkpoint if isinstance(checkpoint, list) else [checkpoint]
    models = [model_class(config, dataset).to(device) for _ in range(len(state_dicts))]
    for i, model in enumerate(models):
        model.load_state_dict(state_dicts[i])
        model.eval()  # Set to evaluation mode
    
    return models

    
class Decoder(nn.Module):
    def __init__(self, emb_models):
        super().__init__()
        self.emb_models = emb_models          # for .embed_latent()
        self.decoder = MaskedAutoencoder()

    def forward(self, x):
        # x: (B, 576)
        with torch.no_grad():
            x = self.emb_models[0].embed_latent(x)  # still (B, 576)

        # ---- reshape to 4‑D feature map ----
        x = self.latent2map(x)          # (B, 32768)
        x = x.view(-1, 128, 16, 16)     # (B, 128, 16, 16)

        # ---- decode ----
        x = self.decoder(x)             # (B, 1, 256, 256)

        # ---- crop to 228 × 198 ----
        h_start = (256 - 228) // 2      # 14
        w_start = (256 - 198) // 2      # 29
        x = x[:, :, h_start:h_start+228, w_start:w_start+198]

        return x                
    
### ----- ###
### UTILS ###
### ----- ###

def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    batch = next(iter(loader))
    batch = batch.to(device)
    recon = model(batch)
    loss = ae_loss(batch, recon)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item(), recon, batch

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    for batch in loader:
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

    emb_models = load_ebm_model(len(train_ds))
    model = Decoder(emb_models).to(device)
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
                batch_np = eval_batch.squeeze(1).cpu().numpy()# -> (B, 1, H, W) -> (B,H,W)
                recon_np  = eval_recon.squeeze(1).cpu().numpy()

                per_ssim = [
                    ssim(b, r, data_range=1.0)
                    for b, r in zip(batch_np, recon_np)
                ]
                per_psnr = [
                    psnr(b, r, data_range=1.0)
                    for b, r in zip(batch_np, recon_np)
                ]

                val_ssim = np.mean(per_ssim)
                val_psnr = np.mean(per_psnr)

            neptune_logger.log_metric("eval_ssim", val_ssim)
            neptune_logger.log_metric("eval_psnr", val_psnr)
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