import torch
import torch.optim as optim

import numpy as np
import os
import time
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import logging
from torch.utils.data import DataLoader
from vae_model import VAE, BetaVAE
from src.dataloader import MRI2D, Clevr
from src.config.load_config import load_config

# COPYRIGHT: August og August

# Initialize model based on config
def initialize_model(cfg, device):
    if cfg.model == "VAE":
        model = VAE(
            input_channels=3,
            latent_dim=16
        ).to(device)
    elif cfg.model.name == "Beta_VAE":
        model = BetaVAE(
            input_channels=cfg.model.input_channels,
            latent_dim=cfg.model.latent_dim,
            beta=cfg.model.beta
        ).to(device)
    else:
        raise ValueError(f"Model type {cfg.model.name} not recognized")
    return model

# @hydra.main(version_base=None, config_path="../configs", config_name="config")
def train_model(cfg: DictConfig):
    #print(OmegaConf.to_yaml(cfg))

    # Set seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # Initialize Weights & Biases (wandb)
    # wandb.init(project=cfg.project.name, config=OmegaConf.to_container(cfg, resolve=True))

    # Set device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data using the HDF5 data loader
    # train_loader = load_data(cfg, split="train"
    # val_loader = load_data(cfg, split="val")
    config = load_config(cfg)
    cfg = config
    dataset = MRI2D(config)
    test_dataset = MRI2D(config)
    train_loader = DataLoader(dataset, num_workers=config.data_workers, batch_size=config.batch_size, shuffle=True, pin_memory=False)
    val_loader = DataLoader(test_dataset, num_workers=config.data_workers, batch_size=config.batch_size, shuffle=True, pin_memory=False, drop_last=True)

    # Initialize model and move to device
    model = initialize_model(config, device)   
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    # Training loop
    model.train()
    for epoch in range(cfg.num_epoch):
        train_loss = 0.0

        for x_batch, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.num_epoch}", leave=False):
            x_batch = x_batch.to(device)

            optimizer.zero_grad()
            recon_batch, mu, logvar = model(x_batch)
            loss, recon_loss, kld_loss = model.loss_function(recon_batch, x_batch, mu, logvar)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Log batch-wise metrics to W&B
            # wandb.log({
            #     "epoch": epoch + 1,
            #     "batch_loss": loss.item() / len(x_batch),
            #     "reconstruction_loss": recon_loss.item() / len(x_batch),
            #     "kl_divergence": kld_loss.item() / len(x_batch)
            # })
            logging.info(f"Epoch [{epoch + 1}/{cfg.num_epoch}], Batch Loss: {loss.item() / len(x_batch):.4f}")
            logging.info(f"Reconstruction Loss: {recon_loss.item() / len(x_batch):.4f}")
            logging.info(f"KL Divergence: {kld_loss.item() / len(x_batch):.4f}")
    


        # Validate model on validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, _ in val_loader:
                x_batch = x_batch.to(device)
                recon_batch, mu, logvar = model(x_batch)
                loss, _, _ = model.loss_function(recon_batch, x_batch, mu, logvar)
                val_loss += loss.item()
                
        # Compute average losses
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)
        print(f"Epoch [{epoch + 1}/{cfg.num_epoch}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        # wandb.log({"epoch": epoch + 1, "epoch_train_loss": avg_train_loss, "epoch_val_loss": avg_val_loss})
        config.NeptuneLogger.log_metric("Train loss", avg_train_loss, step=int(epoch+1))
        config.NeptuneLogger.log_metric("Val loss", avg_val_loss, step=int(epoch+1))
        # Switch back to training mode for next epoch
        model.train()

    # Create a checkpoint
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'config': cfg.to_dict()
    }
    
    model_save_path = f"experiments/models/vae_checkpoint_{int(time.time())}.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(checkpoint, model_save_path)
    
    # Upload checkpoint to W&B
    # artifact = wandb.Artifact('vae_checkpoint', type='model')
    # artifact.add_file(model_save_path)
    # wandb.log_artifact(artifact)

    # wandb.finish()

if __name__ == "__main__":
    train_model('src/config/config_vae.yml')