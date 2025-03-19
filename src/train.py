import os
import sys
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from easydict import EasyDict
import numpy as np
from imageio import imwrite, get_writer
import cv2
import argparse
import os.path as osp
import random
import logging
import tempfile
from tqdm import tqdm
from src.comet_models import LatentEBM, ToyEBM, BetaVAE_H, LatentEBM128
from dataloader import Clevr, BrainDataset
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from src.utils.neptune_logger import NeptuneLogger
from src.config.load_config import load_config, Config
import lpips

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')



def average_gradients(models):
    size = float(dist.get_world_size())

    for model in models:
        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
            param.grad.data /= size


def gen_image(latents, config, models, im_neg, im, steps, create_graph=True, idx=None):
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
                    ix = j % config.components
                    energy = models[j % config.components].forward(im_neg, latents[j]) + energy

            im_grad, = torch.autograd.grad([energy.sum()], [im_neg], create_graph=create_graph)

            im_neg = im_neg - config.step_lr * im_grad

            latents = latents

            im_neg = torch.clamp(im_neg, 0, 1)
            im_negs.append(im_neg)
            im_neg = im_neg.detach()
            im_neg.requires_grad_()

    return im_neg, im_negs, im_grad, masks


def ema_model(models, models_ema, mu=0.999):
    for (model, model_ema) in zip(models, models_ema):
        for param, param_ema in zip(model.parameters(), model_ema.parameters()):
            param_ema.data[:] = mu * param_ema.data + (1 - mu) * param.data


def sync_model(models):
    size = float(dist.get_world_size())

    for model in models:
        for param in model.parameters():
            dist.broadcast(param.data, 0)


def init_model(config, dataset):
    models = [LatentEBM(config, dataset).to(config.device) for _ in range(config.ensembles)]
    optimizers = [Adam(model.parameters(), lr=config.lr) for model in models]
    return models, optimizers


# def test(train_dataloader, models, config, step=0):
#     # if config.cuda:
#     #     dev = torch.device("cuda")
#     # else:
#     dev = torch.device("cpu")

#     replay_buffer = None

#     [model.eval() for model in models]
#     logging.info('Testing model')
#     for im, idx in tqdm(train_dataloader):

#         im = im.to(dev)
#         idx = idx.to(dev)
#         im = im[:config.batch_size]
#         idx = idx[:config.batch_size]
#         batch_size = im.size(0)
#         latent = models[0].embed_latent(im)

#         latents = torch.chunk(latent, config.components, dim=1)

#         im_init = torch.rand_like(im)
#         assert len(latents) == config.components
#         im_neg, _, im_grad, mask = gen_image(latents, config, models, im_init, im, config.steps, sample=config.sample, 
#                                        create_graph=False)
#         im_neg = im_neg.detach()
#         im_components = []

#         if config.components > 1:
#             for i, latent in enumerate(latents):
#                 im_init = torch.rand_like(im)
#                 latents_select = latents[i:i+1]
#                 im_component, _, _, _ = gen_image(latents_select, config, models, im_init, im, config.steps, sample=config.sample,
#                                            create_graph=False)
#                 im_components.append(im_component)

#             im_init = torch.rand_like(im)
#             latents_perm = [torch.cat([latent[i:], latent[:i]], dim=0) for i, latent in enumerate(latents)]
#             im_neg_perm, _, im_grad_perm, _ = gen_image(latents_perm, config, models, im_init, im, config.steps, sample=config.sample,
#                                                      create_graph=False)
#             im_neg_perm = im_neg_perm.detach()
#             im_init = torch.rand_like(im)
#             add_latents = list(latents)
#             im_neg_additional, _, _, _ = gen_image(tuple(add_latents), config, models, im_init, im, config.steps, sample=config.sample,
#                                                      create_graph=False)

#         im.requires_grad = True
#         im_grads = []

#         for i, latent in enumerate(latents):
#             if config.decoder:
#                 im_grad = torch.zeros_like(im)
#             else:
#                 energy_pos = models[i].forward(im, latents[i])
#                 im_grad = torch.autograd.grad([energy_pos.sum()], [im])[0]
#             im_grads.append(im_grad)

#         im_grad = torch.stack(im_grads, dim=1)

#         s = im.size()
#         im_size = s[-1]

#         im_grad = im_grad.view(batch_size, config.components, 3, im_size, im_size) # [4, 3, 3, 128, 128]
#         im_grad_dense = im_grad.view(batch_size, config.components, 1, 3 * im_size * im_size, 1) # [4, 3, 1, 49152, 1]
#         im_grad_min = im_grad_dense.min(dim=3, keepdim=True)[0]
#         im_grad_max = im_grad_dense.max(dim=3, keepdim=True)[0] # [4, 3, 1, 1, 1]

#         # print(f'im grad shape: {im_grad.shape}') # with clevr: ([16, 2, 3, 64, 64]), with MRI2D: ([16, 2, 3, 228, 198])
#         # im_grad_reshaped = im_grad.view(im_grad.shape[0], im_grad.shape[1]*im_grad.shape[2], im_grad.shape[3], im_grad.shape[4]) # dimension 4 instead of 5 before reducing
#         # x_resized = F.interpolate(im_grad_reshaped, size=(64, 64), mode='bilinear', align_corners=False)
#         # # Reshape back to [16, 2, 3, 64, 64]
#         # im_grad = x_resized.view(16, config.components, 3, 64, 64)
#         # print(f'im grad shape after resizing: {im_grad.shape}')

#         im_grad = (im_grad - im_grad_min) / (im_grad_max - im_grad_min + 1e-5) # [4, 3, 3, 128, 128]
#         im_grad[:, :, :, :1, :] = 1
#         im_grad[:, :, :, -1:, :] = 1
#         im_grad[:, :, :, :, :1] = 1
#         im_grad[:, :, :, :, -1:] = 1
#         im_output = im_grad.permute(0, 3, 1, 4, 2).reshape(batch_size * im_size, config.components * im_size, 3)
#         im_output = im_output.cpu().detach().numpy() * 100

#         im_output = (im_output - im_output.min()) / (im_output.max() - im_output.min())

#         im = im.cpu().detach().numpy().transpose((0, 2, 3, 1)).reshape(batch_size*im_size, im_size, 3)

#         im_output = np.concatenate([im_output, im], axis=1)
#         im_output = im_output*255

#         im_output = im_output.astype(np.uint8)
#         imwrite("result/%s/s%08d_grad.png" % (config.run_name,step), im_output)

#         im_neg = im_neg_tensor = im_neg.detach().cpu()
#         im_components = [im_components[i].detach().cpu() for i in range(len(im_components))]
#         im_neg = torch.cat([im_neg] + im_components)
#         im_neg = np.clip(im_neg, 0.0, 1.0)
#         im_neg = make_grid(im_neg, nrow=int(im_neg.shape[0] / (config.components + 1))).permute(1, 2, 0)
#         im_neg = im_neg.numpy()*255

#         im_neg = im_neg.astype(np.uint8)
#         imwrite("result/%s/s%08d_gen.png" % (config.run_name,step), im_neg)

#         if config.components > 1:
#             im_neg_perm = im_neg_perm.detach().cpu()
#             im_components_perm = []
#             for i,im_component in enumerate(im_components):
#                 im_components_perm.append(torch.cat([im_component[i:], im_component[:i]]))
#             im_neg_perm = torch.cat([im_neg_perm] + im_components_perm)
#             im_neg_perm = np.clip(im_neg_perm, 0.0, 1.0)
#             im_neg_perm = make_grid(im_neg_perm, nrow=int(im_neg_perm.shape[0] / (config.components + 1))).permute(1, 2, 0)
#             im_neg_perm = im_neg_perm.numpy()*255
#             im_neg_perm = im_neg_perm.astype(np.uint8)
#             imwrite("result/%s/s%08d_gen_perm.png" % (config.run_name,step), im_neg_perm)

#             im_neg_additional = im_neg_additional.detach().cpu()
#             im_neg_additional = torch.cat([im_neg_additional] + im_components)
#             im_neg_additional = np.clip(im_neg_additional, 0.0, 1.0)
#             im_neg_additional = make_grid(im_neg_additional, 
#                                 nrow=int(im_neg_additional.shape[0] / (config.components + 1))).permute(1, 2, 0)
#             im_neg_additional = im_neg_additional.numpy()*255
#             im_neg_additional = im_neg_additional.astype(np.uint8)
#             imwrite("result/%s/s%08d_gen_add.png" % (config.run_name,step), im_neg_additional)

#             print('test at step %d done!' % step)
#         break

#     [model.train() for model in models]


def train(train_dataloader, test_dataloader, logger, models, optimizers, config, logdir, rank_idx): # TODO: remove logdir
    # TODO: remove rank_idx as arg
    # it = config.resume_iter # TODO: remove line
    [optimizer.zero_grad() for optimizer in optimizers]

    dev = torch.device("cpu")

    # Use LPIPS loss for CelebA-HQ 128x128
    if config.dataset == "celebahq_128":
        #import lpips # uncomment for celebahq_128
        loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()

    for it in tqdm(range(config.num_epoch)): # TODO: it instead of epoch
        for im, idx in train_dataloader:

            im = im.to(dev)
            idx = idx.to(dev)
            im_orig = im

            random_idx = random.randint(0, config.ensembles - 1)
            random_idx = 0

            latent = models[0].embed_latent(im)

            latents = torch.chunk(latent, config.components, dim=1)

            im_neg = torch.rand_like(im)
            im_neg_init = im_neg

            im_neg, im_negs, im_grad, _ = gen_image(latents, config, models, im_neg, im, config.steps)

            im_negs = torch.stack(im_negs, dim=1)

            energy_pos = 0
            energy_neg = 0

            energy_poss = []
            energy_negs = []
            for i in range(config.components):
                energy_poss.append(models[i].forward(im, latents[i]))
                energy_negs.append(models[i].forward(im_neg.detach(), latents[i]))

            energy_pos = torch.stack(energy_poss, dim=1)
            energy_neg = torch.stack(energy_negs, dim=1)
            ml_loss = (energy_pos - energy_neg).mean()

            im_loss = torch.pow(im_negs[:, -1:] - im[:, None], 2).mean()

            if it < 10000 or config.dataset != "celebahq_128":
                loss = im_loss
            else:
                vgg_loss = loss_fn_vgg(im_negs[:, -1], im).mean()
                loss = vgg_loss  + 0.1 * im_loss

            loss.backward()
            # TODO: normalize loss by batch size?
            # TODO: how is loss computed? Is it correct?
            config.NeptuneLogger.log_metric("im_loss", im_loss, step=int(it))
            config.NeptuneLogger.log_metric("ml_loss", ml_loss, step=int(it))
            config.NeptuneLogger.log_metric("loss", loss, step=int(it))
            # if config.gpus > 1:
            #     average_gradients(models)

            [torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0) for model in models]
            [optimizer.step() for optimizer in optimizers]
            [optimizer.zero_grad() for optimizer in optimizers]

            # if it % config.log_interval == 0 and rank_idx == 0:
            #     loss = loss.item()
            #     energy_pos_mean = energy_pos.mean().item()
            #     energy_neg_mean = energy_neg.mean().item()
            #     energy_pos_std = energy_pos.std().item()
            #     energy_neg_std = energy_neg.std().item()

            #     kvs = {}
            #     kvs['loss'] = loss
            #     kvs['ml_loss'] = ml_loss.item()
            #     kvs['im_loss'] = im_loss.item()

            #     if config.dataset == "celebahq_128" and ('vgg_loss' in kvs):
            #         kvs['vgg_loss'] = vgg_loss.item()

            #     kvs['energy_pos_mean'] = energy_pos_mean
            #     kvs['energy_neg_mean'] = energy_neg_mean
            #     kvs['energy_pos_std'] = energy_pos_std
            #     kvs['energy_neg_std'] = energy_neg_std
            #     kvs['average_im_grad'] = torch.abs(im_grad).max()

            #     string = "Iteration {} ".format(it)

            #     for k, v in kvs.items():
            #         string += "%s: %.6f  " % (k,v)
            #         logger.add_scalar(k, v, it)

            #     print(string)

            if it % config.save_interval == 0 and rank_idx == 0: # TODO: remove save_interval from config and rank_idx
                model_path = osp.join(logdir, "model_{}.pth".format(it))


                ckpt = {'config': config.to_dict()}

                for i in range(len(models)):
                    ckpt['model_state_dict_{}'.format(i)] = models[i].state_dict()

                for i in range(len(optimizers)):
                    ckpt['optimizer_state_dict_{}'.format(i)] = optimizers[i].state_dict()

                logging.info(f'model_path: {model_path}')
                torch.save(ckpt, model_path)
                with tempfile.TemporaryDirectory() as tempdir:
                    config.NeptuneLogger.log_model(file_path=osp.join(tempdir, "model_{}.pth".format(it)), file_name="model_it{}".format(it))
                print("Saving model in directory....")
                print('run test')

                # test(test_dataloader, models, config, step=it)

            it += 1



def main_single(rank, config):
    """rank is number of gpus"""
    rank_idx = 0 * 0 + rank # TODO: remove rank_idx from this file

    if not os.path.exists('result/%s' % config.run_name):
        try:
            os.makedirs('result/%s' % config.run_name)
        except:
            pass

    if config.dataset == 'MRI':
        dataset = BrainDataset(config, train=True)
        test_dataset = BrainDataset(config, train=False)

    elif config.dataset == "clevr":
        dataset = Clevr(config)
        test_dataset = dataset

    print(f'Train dataset has {len(dataset)} samples')
    print(f'Test dataset has {len(test_dataset)} samples')

    shuffle=True
    sampler = None

    # if world_size > 1:
    #     group = dist.init_process_group(backend='nccl', init_method='tcp://localhost:8113', world_size=world_size, rank=rank_idx, group_name="default")

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    device = torch.device('cpu')

    logdir = osp.join(config.logdir, config.run_name)


    # TODO: remove below, dont need to load ecisiting model in train!!
    # if config.resume_iter != 0:
    #     logging.info(f'A pretrained model is loaded since resume_iter = {config.resume_iter}')
    #     model_path = osp.join(logdir, "model_{}.pth".format(config.resume_iter))

    #     checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    #     config = checkpoint['config']

    #     config.resume_iter = config_OLD.resume_iter
    #     config.save_interval = config_OLD.save_interval
    #     config.nodes = config_OLD.nodes
    #     config.train = config_OLD.train
    #     config.batch_size = config_OLD.batch_size
    #     config.decoder = config_OLD.decoder
    #     config.optimize_test = config_OLD.optimize_test
    #     config.temporal = config_OLD.temporal
    #     config.sim = config_OLD.sim
    #     config.run_name = config_OLD.exp
    #     config.step_lr = config_OLD.step_lr
    #     config.steps = config_OLD.steps
    #     config.vae_beta = config_OLD.vae_beta

    #     models, optimizers  = init_model(config, dataset)
    #     state_dict = models[0].state_dict()

    #     for i, (model, optimizer) in enumerate(zip(models, optimizers)):
    #         model.load_state_dict(checkpoint['model_state_dict_{}'.format(i)], strict=False)
    #         optimizer.load_state_dict(checkpoint['optimizer_state_dict_{}'.format(i)], strict=False)

    # else:
    models, optimizers = init_model(config, dataset)

    # if config.gpus > 1:
    #     sync_model(models)

    # if config.dataset == "multidsprites":
    #     train_dataloader = MultiDspritesLoader(config.batch_size)
    #     test_dataloader = MultiDspritesLoader(config.batch_size)
    # elif config.dataset == "tetris":
    #     train_dataloader = TetrominoesLoader(config.batch_size)
    #     test_dataloader = TetrominoesLoader(config.batch_size)

    train_dataloader = DataLoader(dataset, num_workers=config.data_workers, batch_size=config.batch_size, shuffle=shuffle, pin_memory=False)
    test_dataloader = DataLoader(test_dataset, num_workers=config.data_workers, batch_size=config.batch_size, shuffle=True, pin_memory=False, drop_last=True)

    print(f'Train dataloader has {len(train_dataloader)} batches')
    print(f'Test dataloader has {len(test_dataloader)} batches')

    logger = SummaryWriter(logdir)

    # print(f'config.train: {config.train}')
    # logging.info(f'config: {config}')

    models = [model.train() for model in models]
    train(train_dataloader, test_dataloader, logger, models, optimizers, config, logdir, rank_idx)
    # else:
    #     models = [model.eval() for model in models]

        

    # elif config.optimize_test:
    #     test_optimize(test_dataloader, models, config, step=config.resume_iter)
    # else:
    #     test(test_dataloader, models, config)


def main(config: Config):
    # mp.spawn(main_single, nprocs=FLAGS.gpus, args=(FLAGS,))
    main_single(0, config)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, help="Path to config file", default='src/config/test.yml')
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)

