import os
import sys
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler
import argparse
import logging
from tqdm import tqdm
from src.comet_models import LatentEBM, LatentEBM128
from dataloader import Clevr, BrainDataset, MRI2D
import threading
from dataclasses import asdict
from src.utils.neptune_logger import NeptuneLogger
from torch.cuda.amp import autocast, GradScaler
from piqa import SSIM                                   # differentiable SSIM
from torchvision.utils import make_grid, save_image
import math
from PIL import Image

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from src.config.load_config import load_config, Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

logging.info("Importing log this")

def mse_psnr(x, y):
    mse = F.mse_loss(x, y).item()
    return mse, 20 * math.log10(1. / math.sqrt(mse + 1e-12))

def get_ssim(nc):
    ssim = SSIM(n_channels=nc)
    return ssim.cuda() if torch.cuda.is_available() else ssim

def gen_image(latents, config, models, im_neg, im, steps = 100, create_graph=True, idx=None):
    # TODO: the samples were used through langevin, where did they go?
    # TODO: optimal number of steps?
    im_noise = torch.randn_like(im_neg).detach()

    im_negs = []
    latents = torch.stack(latents, dim=0)

    if config.decoder: # TODO: split genimage into a with and without decoder functiom
        masks = []
        colors = []
        for i in range(len(latents)):
            if idx is not None and idx != i:
                pass
            else:
                color, mask = models[i % config.components].forward(None, latents[i])
                masks.append(mask)
                colors.append(color)
        masks = F.softmax(torch.stack(masks, dim=1), dim=1)
        colors = torch.stack(colors, dim=1)
        im_neg = torch.sum(masks * colors, dim=1)
        im_negs = [im_neg]
        im_grad = torch.zeros_like(im_neg)
    else:
        im_neg.requires_grad_(requires_grad=True)
        s = im.size()
        masks = torch.zeros(s[0], config.components, s[-2], s[-1]).to(im_neg.device)
        masks.requires_grad_(requires_grad=True)

        for i in range(steps):
            im_noise.normal_()

            energy = 0
            for j in range(len(latents)):
                if idx is not None and idx != j:
                    pass
                else:
                    energy = models[j % config.components].forward(im_neg, latents[j]) + energy

            # this autograd causes memory issues, the tensor in comet models cant be saved
            im_grad, = torch.autograd.grad([energy.sum()], [im_neg], create_graph=create_graph)

            im_neg = im_neg - config.step_lr * im_grad

            latents = latents

            im_neg = torch.clamp(im_neg, 0, 1)
            # if not last step
            if i != steps - 1:
                im_neg = im_neg.detach()
                im_neg.requires_grad_()
            else:
                im_negs.append(im_neg)

    return im_neg, im_negs, im_grad, masks


def init_model(config, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.model == 'LatentEBM':
        models = [LatentEBM(config, dataset).to(device) for _ in range(config.ensembles)]
    else:
        models = [LatentEBM128(config, dataset).to(device) for _ in range(config.ensembles)]
    # TODO? enseblemes should be == components
    # TODO: we should make some runs where we opitmize lr, steps, batches etc.
    optimizers = [Adam(model.parameters(), lr=config.lr) for model in models]
    
    # Simple scheduler that reduces LR by 0.9 every 1/10th of training steps
    step_size = max(1, config.steps // 10)
    schedulers = [torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.8) for optimizer in optimizers]
    
    return models, optimizers, schedulers


def train(train_dataloader, models, optimizers, schedulers, config, neptune_logger: NeptuneLogger):
    # Log the configuration
    neptune_logger.log_config_dict(asdict(config))

    if torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    # Initialize GradScaler for mixed precision
    scaler = GradScaler()

    for it, im in tqdm(enumerate(train_dataloader), total=config.steps):
        im = im.to(dev)  # 8, 3, 64, 64

        optimizers[0].zero_grad()

        with autocast():  # Updated autocast usage
            latent = models[0].embed_latent(im)   # vector latent
            latents = torch.chunk(latent, config.components, dim=1)
            im_neg = torch.rand_like(im)
            im_neg, im_negs, _, _ = gen_image(latents, config, models, im_neg, im)
            
            # Fix the stacking and reshaping for make_grid
            if len(im_negs) > 1:
                im_negs = torch.stack(im_negs, dim=1)  # (B, num_steps, C, H, W)
                # Reshape for make_grid: (B * num_steps, C, H, W)
                b, num_steps, c, h, w = im_negs.shape
                im_negs_for_grid = im_negs.reshape(b * num_steps, c, h, w)
            else:
                # If only one image, just use it directly
                im_negs_for_grid = im_negs[0]
                im_negs = torch.stack(im_negs, dim=1)  # Keep for loss computation
            
            im_loss = torch.pow(im_negs[:, -1:] - im[:, None], 2).mean()
            loss = im_loss

        neptune_logger.log_metric("im_loss", im_loss.item(), step=int(it))
        neptune_logger.log_metric("loss", loss.item(), step=int(it))
        
        # Use the properly shaped tensor for make_grid and format for Neptune
        im_negs_grid = make_grid(im_negs_for_grid, nrow=min(config.components, im_negs_for_grid.size(0)))
        
        # Convert from (C, H, W) to (H, W, C) for Neptune and ensure it's in the right format
        if im_negs_grid.dim() == 3:
            im_negs_grid = im_negs_grid.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
        
        # Convert to CPU and clamp values to [0, 1]
        im_negs_grid = torch.clamp(im_negs_grid, 0, 1).cpu()
        
        # If single channel, squeeze the last dimension for grayscale
        if im_negs_grid.shape[-1] == 1:
            im_negs_grid = im_negs_grid.squeeze(-1)  # (H, W, 1) -> (H, W)
        
        neptune_logger.log_image("im_negs", im_negs_grid, step=int(it))

        # Backpropagation with GradScaler
        scaler.scale(loss).backward()

        # Optimizer step with GradScaler
        scaler.step(optimizers[0])
        scaler.update()

        # Scheduler step - FIX: StepLR doesn't take loss argument, and only step the scheduler for the optimizer being used
        [scheduler.step() for scheduler in schedulers]
        
        # Log learning rate AFTER scheduler step to see the updated value
        neptune_logger.log_metric("scheduler_lr", optimizers[0].param_groups[0]['lr'], step=int(it))
        neptune_logger.log_metric("scheduler_lr2", optimizers[1].param_groups[0]['lr'], step=int(it))
        neptune_logger.log_metric("scheduler_lr3", optimizers[2].param_groups[0]['lr'], step=int(it))

        if it % 100 == 0:
            torch.save({"it": it,
                        "ebm_state": [m.state_dict() for m in models]},
                        f"{config.run_dir}/best_model1.pt")
            neptune_logger.log_model(f"{config.run_dir}/best_model1.pt", f"models_{it}.pth")


def main(config: Config, neptune_logger: NeptuneLogger):
    if config.dataset == 'MRI':
        dataset = BrainDataset(config, train=True)
        test_dataset = BrainDataset(config, train=False)
    elif config.dataset == "clevr":
        dataset = Clevr(config, train=True)
        test_dataset = Clevr(config, train=False)
    elif config.dataset == '2DMRI':
        dataset = MRI2D(config) # TOOD: test and train cannor be the same
        test_dataset = MRI2D(config, eval=True)

    print(f'Train dataset has {len(dataset)} samples')
    print(f'Test dataset has {len(test_dataset)} samples')

    models, optimizers, schedulers = init_model(config, dataset)

    random_sampler = RandomSampler(dataset, replacement=True, num_samples=config.steps * config.batch_size) 
    train_dataloader = DataLoader(dataset, num_workers=config.data_workers, batch_size=config.batch_size, sampler=random_sampler, pin_memory=False)
    test_dataloader = DataLoader(test_dataset, num_workers=config.data_workers, batch_size=config.batch_size, shuffle=True, pin_memory=False, drop_last=True)

    print(f'Train dataloader has {len(train_dataloader)} batches')
    print(f'Test dataloader has {len(test_dataloader)} batches')

    logging.info(f'config: {config}')
    models = [model.train() for model in models]
    train(train_dataloader, models, optimizers, schedulers, config, neptune_logger)


def listen_for_exit():
    while True:
        if input().strip().lower() == 'q':
            logging.info("Exiting script...")
            os._exit(0)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, help="Path to config file", default='src/config/2DMRI_config.yml') # src/config/test.yml
    args = parser.parse_args()

    config = load_config(args.config)
    neptune_logger = NeptuneLogger(test=False, description=config.run_name)

    exit_listener = threading.Thread(target=listen_for_exit, daemon=True)
    exit_listener.start()

    main(config, neptune_logger)

    # TODO: consider gradient checkpointing to reduce memory usage,
    # TODO: or use AMP
    # *TODO: oprimal batch size??