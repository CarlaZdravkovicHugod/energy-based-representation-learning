import torch
from typing import List
from comet_models import LatentEBM
import torch.nn.functional as F
import logging
from torch.optim import Adam
from torch.utils.data import DataLoader
import os.path as osp
import numpy as np
from imageio import imwrite
import argparse
from torchvision.utils import make_grid
from tqdm import tqdm
import tempfile

from src.config.load_config import load_config, Config
from src.dataloader import BrainDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

def gen_image_with_decoder(latents, config, models):
    """
    Generates an image using a learned model with a decoder by optimizing the image to minimize its energy.
    
    Args:
        latents (list): List of latent vectors.
        config (Config): Configuration object.
        models (list): List of models.
    
    Returns:
        Tuple: Generated image, intermediate images, gradients, and masks.
    """
    masks, colors = [], []
    for i, latent in enumerate(latents):
        color, mask = models[i % config.components].forward(None, latent)
        masks.append(mask)
        colors.append(color)
    masks = F.softmax(torch.stack(masks, dim=1), dim=1)
    colors = torch.stack(colors, dim=1)
    im_neg = torch.sum(masks * colors, dim=1)
    im_negs = [im_neg]
    im_grad = torch.zeros_like(im_neg)
    return im_neg, im_negs, im_grad, masks


def gen_image(latents, config, models, im_neg, im, num_steps, create_graph=True, idx=None):
    """
    Generates an image using a learned model by optimizing the image to minimize its energy.
    
    Args:
        latents (list): List of latent vectors.
        config (Config): Configuration object.
        models (list): List of models.
        im_neg (Tensor): Initial negative image.
        im (Tensor): Target image.
        num_steps (int): Number of optimization steps.
        create_graph (bool): Whether to create a computational graph.
        idx (int, optional): Index for selecting specific latent vectors.
    
    Returns:
        Tuple: Generated image, intermediate images, gradients, and masks.
    """
    im_noise = torch.randn_like(im_neg).detach()
    im_negs = []
    latents = torch.stack(latents, dim=0)

    im_neg.requires_grad_(True)
    s = im.size()
    masks = torch.zeros(s[0], config.components, s[-2], s[-1], device=im_neg.device)
    masks.requires_grad_(True)

    for i in range(num_steps):
        im_noise.normal_()
        energy = sum(models[j % len(models)].forward(im_neg, latents[j]) for j in range(len(latents)) if idx is None or idx == j) # using len(models) instead of number of components
        im_grad, = torch.autograd.grad([energy.sum()], [im_neg], create_graph=False) # TODO: change to true if needed
        im_neg = torch.clamp(im_neg - config.step_lr * im_grad, 0, 1)
        im_negs.append(im_neg.detach())
        im_neg.requires_grad_()

    return im_neg, im_negs, im_grad, masks


def init_model(config: Config, dataset: BrainDataset):
    """ Initializes the model and optimizer.
    Creates multiple LatentEBM models based on config.ensembles.
    Moves models to the configured device (CPU/GPU).
    Initializes optimizers (Adam) for each model.
    Args:
        config (Config): Configuration object.
        dataset (BrainDataset): Dataset object.
    Returns:
        models (list): List of models.
        optimizers (list): List of optimizers.
    """
    models = [LatentEBM(config, dataset).to(config.device) for _ in range(config.ensembles)]
    optimizers = [Adam(model.parameters(), lr=config.lr) for model in models]
    return models, optimizers


def test(train_dataloader, models, config: Config, step=0): # NOT REFACTORED
    """Tests the model by generating images using the learned model.
    The key steps are:
    1. Set the model to evaluation mode.
    2. Generate latent embeddings for the input images.
    3. Generate images using the learned model.
    4. Compute gradients of the images.
    5. Save the generated images and gradients.
    Args:
        train_dataloader (DataLoader): DataLoader object for training dataset.
        models (list): List of models to test.
        config (Config): Configuration object.
        step (int): Current step of training.
    Returns:
        None
    """
    [model.eval() for model in models]
    for im, idx in train_dataloader:

        im = im.to(config.device)
        idx = idx.to(config.device)
        im = im[:config.num_visuals]
        idx = idx[:config.num_visuals]
        batch_size = im.size(0)
        latent = models[0].embed_latent(im)

        latents = torch.chunk(latent, config.components, dim=1)

        im_init = torch.rand_like(im)
        assert len(latents) == config.components
        im_neg, _, im_grad, mask = gen_image(latents, config, models, im_init, im, create_graph=False)
        im_neg = im_neg.detach()
        im_components = []

        if config.components > 1:
            for i, latent in enumerate(latents):
                im_init = torch.rand_like(im)
                latents_select = latents[i:i+1]
                im_component, _, _, _ = gen_image(latents_select, config, models, im_init, im, create_graph=False)
                im_components.append(im_component)

            im_init = torch.rand_like(im)
            latents_perm = [torch.cat([latent[i:], latent[:i]], dim=0) for i, latent in enumerate(latents)]
            im_neg_perm, _, im_grad_perm, _ = gen_image(latents_perm, config, models, im_init, im, create_graph=False)
            im_neg_perm = im_neg_perm.detach()
            im_init = torch.rand_like(im)
            add_latents = list(latents)
            for i in range(config.num_additional):
                add_latents.append(torch.roll(latents[i], i + 1, 0))
            im_neg_additional, _, _, _ = gen_image(tuple(add_latents), config, models, im_init, im, create_graph=False)

        im.requires_grad = True
        im_grads = []

        for i, latent in enumerate(latents):
            if config.decoder:
                im_grad = torch.zeros_like(im)
            else:
                energy_pos = models[i].forward(im, latents[i])
                im_grad = torch.autograd.grad([energy_pos.sum()], [im])[0]
            im_grads.append(im_grad)

        im_grad = torch.stack(im_grads, dim=1)

        s = im.size()
        im_size = s[-1]

        im_grad = im_grad.view(batch_size, config.components, 3, im_size, im_size) # [4, 3, 3, 128, 128]
        im_grad_dense = im_grad.view(batch_size, config.components, 1, 3 * im_size * im_size, 1) # [4, 3, 1, 49152, 1]
        im_grad_min = im_grad_dense.min(dim=3, keepdim=True)[0]
        im_grad_max = im_grad_dense.max(dim=3, keepdim=True)[0] # [4, 3, 1, 1, 1]

        im_grad = (im_grad - im_grad_min) / (im_grad_max - im_grad_min + 1e-5) # [4, 3, 3, 128, 128]
        im_grad[:, :, :, :1, :] = 1
        im_grad[:, :, :, -1:, :] = 1
        im_grad[:, :, :, :, :1] = 1
        im_grad[:, :, :, :, -1:] = 1
        im_output = im_grad.permute(0, 3, 1, 4, 2).reshape(batch_size * im_size, config.components * im_size, 3)
        im_output = im_output.cpu().detach().numpy() * 100

        im_output = (im_output - im_output.min()) / (im_output.max() - im_output.min())

        im = im.cpu().detach().numpy().transpose((0, 2, 3, 1)).reshape(batch_size*im_size, im_size, 3)

        im_output = np.concatenate([im_output, im], axis=1)
        im_output = im_output*255
        imwrite("result/%s/s%08d_grad.png" % (config.exp,step), im_output)

        im_neg = im_neg_tensor = im_neg.detach().cpu()
        im_components = [im_components[i].detach().cpu() for i in range(len(im_components))]
        im_neg = torch.cat([im_neg] + im_components)
        im_neg = np.clip(im_neg, 0.0, 1.0)
        im_neg = make_grid(im_neg, nrow=int(im_neg.shape[0] / (config.components + 1))).permute(1, 2, 0)
        im_neg = im_neg.numpy()*255
        imwrite("result/%s/s%08d_gen.png" % (config.exp,step), im_neg)

        if config.components > 1:
            im_neg_perm = im_neg_perm.detach().cpu()
            im_components_perm = []
            for i,im_component in enumerate(im_components):
                im_components_perm.append(torch.cat([im_component[i:], im_component[:i]]))
            im_neg_perm = torch.cat([im_neg_perm] + im_components_perm)
            im_neg_perm = np.clip(im_neg_perm, 0.0, 1.0)
            im_neg_perm = make_grid(im_neg_perm, nrow=int(im_neg_perm.shape[0] / (config.components + 1))).permute(1, 2, 0)
            im_neg_perm = im_neg_perm.numpy()*255
            imwrite("result/%s/s%08d_gen_perm.png" % (config.exp,step), im_neg_perm)

            im_neg_additional = im_neg_additional.detach().cpu()
            for i in range(config.num_additional):
                im_components.append(torch.roll(im_components[i], i + 1, 0))
            im_neg_additional = torch.cat([im_neg_additional] + im_components)
            im_neg_additional = np.clip(im_neg_additional, 0.0, 1.0)
            im_neg_additional = make_grid(im_neg_additional,
                                nrow=int(im_neg_additional.shape[0] / (config.components + config.num_additional + 1))).permute(1, 2, 0)
            im_neg_additional = im_neg_additional.numpy()*255
            imwrite("result/%s/s%08d_gen_add.png" % (config.exp,step), im_neg_additional)

            logging.info('Test at step %d done!' % step)
        break

    [model.train() for model in models]


def train(train_dataloader, test_dataloader, models: List[LatentEBM], optimizers: List[Adam], config: Config):
    """Trains the model using the training dataset.
    The key steps are:
    1. Initialize the model and optimizer.
    2. Iterate over the training dataset.
    3. Generate negative samples using the learned model.
    4. Compute the energy of the positive and negative samples.
    5. Compute the loss and backpropagate the gradients.
    6. Save the model checkpoint.

    Args:
        train_dataloader (DataLoader): DataLoader object for training dataset.
        test_dataloader (DataLoader): DataLoader object for testing dataset.
        models (list): List of models to train.
        optimizers (list): List of optimizers for each model.
        config (Config): Configuration object.
    Returns:
        None
    """
    #it = 0 # no var resum_iter in config,
    [optimizer.zero_grad() for optimizer in optimizers]

    for it in tqdm(range(config.num_epoch)):
        for idx,im in enumerate(train_dataloader):

            im = im.to(config.device)
            #idx = idx.to(config.device) # you cant do that on an int, such as index, not sure why it would make sense either
            im = im.unsqueeze(1).repeat(1,3,1,1) # add channel dimension (RGB = 3 channels)
            latent = models[0].embed_latent(im)
            latents = torch.chunk(latent, config.components, dim=1)

            im_neg = torch.rand_like(im)
            logging.info('Generating image')
            im_neg, im_negs, im_grad, _ = gen_image(latents, config, models, im_neg, im,num_steps=1) # TODO: choose number of optimization steps

            im_negs = torch.stack(im_negs, dim=1)

            energy_poss = []
            energy_negs = []

            for i in range(config.components): # i % config.ensembles may be wrong
                energy_poss.append(models[i % config.ensembles].forward(im, latents[i]))
                energy_negs.append(models[i % config.ensembles].forward(im_neg.detach(), latents[i]))

            energy_pos = torch.stack(energy_poss, dim=1)
            energy_neg = torch.stack(energy_negs, dim=1)
            ml_loss = (energy_pos - energy_neg).mean()

            im_loss = torch.pow(im_negs[:, -1:] - im[:, None], 2).mean()
            loss = im_loss + 0.1 * ml_loss
            loss.backward()
            logging.info(f'Iteration {it}')
            # TODO: add warning/logging if loss is nan
            config.NeptuneLogger.log_metric("im_loss", im_loss, step=idx) # TODO: check it, should not be 0
            config.NeptuneLogger.log_metric("ml_loss", ml_loss, step=idx)
            config.NeptuneLogger.log_metric("loss", loss, step=idx)

            [torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0) for model in models]
            [optimizer.step() for optimizer in optimizers]
            [optimizer.zero_grad() for optimizer in optimizers]

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = osp.join(temp_dir, "model_{}.pth".format(it))
            # convert to dict
            ckpt = {'config': config.to_dict()}

            for i in range(len(models)):
                ckpt['model_state_dict_{}'.format(i)] = models[i].state_dict()

            for i in range(len(optimizers)):
                ckpt['optimizer_state_dict_{}'.format(i)] = optimizers[i].state_dict()

            torch.save(ckpt, model_path)
            config.NeptuneLogger.log_model(model_path, file_name="model_it{}".format(it)) # TODO: ckeck that it and not idx is true
            logging.info("Saving model in directory....") # TODO: This cant be right, model is saved nump epochs times, so the loop is not working
        logging.info('Now running test...')

        test(test_dataloader, models, config, step=it)


def main_single(config: Config):
    """Main function for training the model.
    The key steps are:
    1. Load the dataset.
    2. Initialize the model and optimizer.
    3. Create DataLoader objects for training and testing datasets.
    4. Train the model.
    5. Test the model.
    Args:
        config (Config): Configuration object.
    Returns:
        None
    """
    dataset = BrainDataset(config).load_data() # BrainDataset(config.data_path)
    train_size = int(config.train_data_size * len(dataset))
    test_size = len(dataset) - train_size
    dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    models, optimizers = init_model(config, dataset)
    models = [model.train() for model in models]

    train_dataloader = DataLoader(dataset, num_workers=config.data_workers, batch_size=config.batch_size, shuffle=config.shuffle_dataset, pin_memory=False)
    test_dataloader = DataLoader(test_dataset, num_workers=config.data_workers, batch_size=config.batch_size, shuffle=True, pin_memory=False, drop_last=True)

    train(train_dataloader, test_dataloader, models, optimizers, config)
    # test(test_dataloader, models, config)


def main(config: Config):
    # mp.spawn(main_single, nprocs=FLAGS.gpus, args=(FLAGS,))
    main_single(config)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, help="Path to config file", default='src/config/test.yml')
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)
