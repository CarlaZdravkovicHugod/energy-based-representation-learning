import torch
from typing import List
from comet.models import LatentEBM, ToyEBM, BetaVAE_H, LatentEBM128
import torch.nn.functional as F
import os
from dataloader import BrainDataset
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os.path as osp
import numpy as np
from imageio import imwrite
import argparse
import random
from torchvision.utils import make_grid
from src.config.load_config import load_config, Config
from tqdm import tqdm
import tempfile


# """Parse input arguments"""
# parser = argparse.ArgumentParser(description='Train EBM model')

# parser.add_argument('--train', action='store_true', help='whether or not to train')
# parser.add_argument('--optimize_test', action='store_true', help='whether or not to train')
# parser.add_argument('--cuda', action='store_true', help='whether to use cuda or not')
# parser.add_argument('--single', action='store_true', help='test overfitting of the dataset')


# parser.add_argument('--dataset', default='blender', type=str, help='Dataset to use (intphys or others or imagenet or cubes)')
# parser.add_argument('--logdir', default='cachedir', type=str, help='location where log of experiments will be stored')
# parser.add_argument('--exp', default='default', type=str, help='name of experiments')

# # training
# parser.add_argument('--resume_iter', default=0, type=int, help='iteration to resume training')
# parser.add_argument('--batch_size', default=64, type=int, help='size of batch of input to use')
# parser.add_argument('--num_epoch', default=10000, type=int, help='number of epochs of training to run')
# parser.add_argument('--lr', default=1e-4, type=float, help='learning rate for training')
# parser.add_argument('--log_interval', default=10, type=int, help='log outputs every so many batches')
# parser.add_argument('--save_interval', default=1000, type=int, help='save outputs every so many batches')

# # data
# parser.add_argument('--data_workers', default=4, type=int, help='Number of different data workers to load data in parallel')
# parser.add_argument('--ensembles', default=1, type=int, help='use an ensemble of models')
# parser.add_argument('--vae-beta', type=float, default=0.)

# # EBM specific settings

# # Model specific settings
# parser.add_argument('--filter_dim', default=64, type=int, help='number of filters to use')
# parser.add_argument('--components', default=2, type=int, help='number of components to explain an image with')
# parser.add_argument('--component_weight', action='store_true', help='optimize for weights of the components also')
# parser.add_argument('--tie_weight', action='store_true', help='tie the weights between seperate models')
# parser.add_argument('--optimize_mask', action='store_true', help='also optimize a segmentation mask over image')
# parser.add_argument('--recurrent_model', action='store_true', help='use a recurrent model to infer latents')
# parser.add_argument('--pos_embed', action='store_true', help='add a positional embedding to model')
# parser.add_argument('--spatial_feat', action='store_true', help='use spatial latents for object segmentation')


# parser.add_argument('--num_steps', default=10, type=int, help='Steps of gradient descent for training')
# parser.add_argument('--num_visuals', default=16, type=int, help='Number of visuals')
# parser.add_argument('--num_additional', default=0, type=int, help='Number of additional components to add')

# parser.add_argument('--step_lr', default=500.0, type=float, help='step size of latents')

# parser.add_argument('--latent_dim', default=64, type=int, help='dimension of the latent')
# parser.add_argument('--sample', action='store_true', help='generate negative samples through Langevin')
# parser.add_argument('--decoder', action='store_true', help='decoder for model')

# # Distributed training hyperparameters
# parser.add_argument('--nodes', default=1, type=int, help='number of nodes for training')
# parser.add_argument('--gpus', default=1, type=int, help='number of gpus per nodes')
# parser.add_argument('--node_rank', default=0, type=int, help='rank of node')

# NOT REFACTORED
def average_gradients(models):
    size = float(dist.get_world_size())

    for model in models:
        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
            param.grad.data /= size

# NOT REFACTORED
def gen_image(latents, FLAGS, models, im_neg, im, num_steps, sample=False, create_graph=True, idx=None, weights=None):
    im_noise = torch.randn_like(im_neg).detach()
    im_negs_samples = []

    im_negs = []

    latents = torch.stack(latents, dim=0)

    if FLAGS.decoder:
        masks = []
        colors = []
        for i in range(len(latents)):
            if idx is not None and idx != i:
                pass
            else:
                color, mask = models[i % FLAGS.components].forward(None, latents[i])
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
        masks = torch.zeros(s[0], FLAGS.components, s[-2], s[-1]).to(im_neg.device)
        masks.requires_grad_(requires_grad=True)

        for i in range(num_steps):
            im_noise.normal_()

            energy = 0
            for j in range(len(latents)):
                if idx is not None and idx != j:
                    pass
                else:
                    ix = j % FLAGS.components
                    energy = models[j % FLAGS.components].forward(im_neg, latents[j]) + energy

            im_grad, = torch.autograd.grad([energy.sum()], [im_neg], create_graph=create_graph)

            im_neg = im_neg - FLAGS.step_lr * im_grad

            latents = latents

            im_neg = torch.clamp(im_neg, 0, 1)
            im_negs.append(im_neg)
            im_neg = im_neg.detach()
            im_neg.requires_grad_()

    return im_neg, im_negs, im_grad, masks


def ema_model(models, models_ema, mu=0.999): # NOT REFACTORED
    for (model, model_ema) in zip(models, models_ema):
        for param, param_ema in zip(model.parameters(), model_ema.parameters()):
            param_ema.data[:] = mu * param_ema.data + (1 - mu) * param.data


def sync_model(models): # NOT REFACTORED
    size = float(dist.get_world_size())

    for model in models:
        for param in model.parameters():
            dist.broadcast(param.data, 0)


def init_model(config: Config, dataset: BrainDataset):
    models = [LatentEBM(config, dataset).to(config.device) for _ in range(config.ensembles)]
    optimizers = [Adam(model.parameters(), lr=config.lr) for model in models]
    return models, optimizers


def test(train_dataloader, models, config: Config, step=0): # NOT REFACTORED
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

            print('test at step %d done!' % step)
        break

    [model.train() for model in models]


def train(train_dataloader, test_dataloader, models: List[LatentEBM], optimizers: List[Adam], config: Config):
    it = config.resume_iter
    [optimizer.zero_grad() for optimizer in optimizers]

    for _ in range(config.num_epoch):
        for it, (im, idx) in enumerate(train_dataloader):

            im = im.to(config.device)
            idx = idx.to(config.device)

            latent = models[0].embed_latent(im)
            latents = torch.chunk(latent, config.components, dim=1)

            im_neg = torch.rand_like(im)
            im_neg, im_negs, im_grad, _ = gen_image(latents, config, models, im_neg, im,)

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
            config.NeptuneLogger.log_metric("im_loss", im_loss, step=it)
            config.NeptuneLogger.log_metric("ml_loss", ml_loss, step=it)
            config.NeptuneLogger.log_metric("loss", loss, step=it)
            
            [torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0) for model in models]
            [optimizer.step() for optimizer in optimizers]
            [optimizer.zero_grad() for optimizer in optimizers]

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = osp.join(temp_dir, "model_{}.pth".format(it))
            ckpt = {'config': config}

            for i in range(len(models)):
                ckpt['model_state_dict_{}'.format(i)] = models[i].state_dict()

            for i in range(len(optimizers)):
                ckpt['optimizer_state_dict_{}'.format(i)] = optimizers[i].state_dict()

            torch.save(ckpt, model_path)
            config.NeptuneLogger.log_model(model_path, name="model_it{}".format(it))
            print("Saving model in directory....")
        print('Now running test...')

        test(test_dataloader, models, config, step=it)


def main_single(config: Config):
    dataset = BrainDataset(config.data_path)
    test_dataset = dataset
    
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
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    main(config)
