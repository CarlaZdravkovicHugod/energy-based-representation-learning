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

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from src.config.load_config import load_config, Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

logging.info("Importing log this")

# TODO: try using latentEBM128

def gen_image(latents, config, models, im_neg, im, steps = 10, create_graph=True, idx=None):
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
            im_negs.append(im_neg)
            im_neg = im_neg.detach()
            im_neg.requires_grad_()

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
    schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=200) for optimizer in optimizers]
    return models, optimizers, schedulers


def train(train_dataloader, models, optimizers, schedulers, config):
    # Log the configuration
    neptune_logger.log_config_dict(asdict(config))

    if torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    # Initialize GradScaler for mixed precision
    scaler = GradScaler()

    for it, (im, idx) in tqdm(enumerate(train_dataloader), total=config.steps):
        im = im.to(dev)  # 8, 3, 64, 64
        idx = idx.to(dev)

        optimizers[0].zero_grad()

        with autocast():  # Enable mixed precision
            latent = models[0].embed_latent(im)
            latents = torch.chunk(latent, config.components, dim=1)
            im_neg = torch.rand_like(im)
            im_neg, im_negs, _, _ = gen_image(latents, config, models, im_neg, im)
            im_negs = torch.stack(im_negs, dim=1)
            im_loss = torch.pow(im_negs[:, -1:] - im[:, None], 2).mean()
            loss = im_loss

        # Log metrics
        neptune_logger.log_metric("loss", loss.item(), step=int(it))
        neptune_logger.log_metric("scheduler_lr", optimizers[0].param_groups[0]['lr'], step=int(it))

        # Backpropagation with GradScaler
        scaler.scale(loss).backward()

        # Optimizer step with GradScaler
        scaler.step(optimizers[0])
        scaler.update()

        # Scheduler step
        [scheduler.step(loss) for scheduler in schedulers]

        # Save model checkpoints periodically
        if it % 100 == 0:
            models_copy = [model.state_dict() for model in models]
            torch.save(models_copy, f"models.pth")
            neptune_logger.log_model(f"models.pth", f"models_{it}.pth")


def main(config: Config, neptune_logger: NeptuneLogger):
    if config.dataset == 'MRI':
        dataset = BrainDataset(config, train=True)
        test_dataset = BrainDataset(config, train=False)
    elif config.dataset == "clevr":
        dataset = Clevr(config, train=True)
        test_dataset = Clevr(config, train=False)
    elif config.dataset == '2DMRI':
        dataset = MRI2D(config) # TOOD: test and train cannor be the same
        test_dataset = MRI2D(config)

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
    train(train_dataloader, models, optimizers, schedulers, config)


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