import matplotlib.pyplot as plt
import torch
from comet_models import LatentEBM, LatentEBM128
from config.load_config import load_config
from dataloader import MRI2D, Clevr
import matplotlib.pyplot as plt
from imageio.v2 import get_writer
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler

# TODO: when ok model from comets train script on clevr data, then test the reconsiuctrion of clevr data
# so we can compare with our own model's performance.


def gen_image(latents, FLAGS, models, im_neg, num_steps, idx=None):
    im_negs = []

    im_neg.requires_grad_(requires_grad=True)

    for i in tqdm(range(num_steps)):
        energy = 0

        for j in range(len(latents)):
            if idx is not None and idx != j:
                pass
            else:
                ix = j % FLAGS.components
                energy = models[j % FLAGS.components].forward(im_neg, latents[j]) + energy

        im_grad, = torch.autograd.grad([energy.sum()], [im_neg])

        im_neg = im_neg - FLAGS.step_lr * im_grad

        im_neg = torch.clamp(im_neg, 0, 1)
        im_negs.append(im_neg)
        im_neg = im_neg.detach()
        im_neg.requires_grad_()

    return im_negs


if __name__ == "__main__":

    config = load_config('/Users/carlahugod/Desktop/UNI/6sem/bach/energy-based-representation-learning/src/config/2DMRI_config.yml')
    dataset = MRI2D(config)
    our_model = False
    clevr = True


    if our_model and not clevr:
        state_dicts = torch.load('/Users/carlahugod/Desktop/UNI/6sem/bach/energy-based-representation-learning/models.pth')
        print(type(state_dicts))
        print(len(state_dicts))
        models = [LatentEBM(config, dataset).cpu() for _ in range(len(state_dicts))]
        for i, model in enumerate(models):
            model.load_state_dict(state_dicts[i])
            model.eval()  # Set to evaluation mode
        img = dataset.__getitem__(0)[0].squeeze(0).numpy()
        print(img.shape)
        latent = models[0].embed_latent(dataset.__getitem__(0)[0])
        latents = torch.chunk(latent, 5, dim=1)

        tensor_img = dataset.__getitem__(0)[0]
        print(tensor_img.shape)
        # add specific batch size 12 to shape:
        tensor_img = tensor_img.unsqueeze(0).expand(12, -1, -1, -1)
        print(tensor_img.shape)

        latent = models[0].embed_latent(tensor_img)
        latents = torch.chunk(latent, config.components, dim=1)
        im_neg = torch.rand_like(tensor_img)
        im_negs = gen_image(latents, config, models, im_neg, 30)


        # Visualize each tensor in im_negs
        plt.figure(figsize=(15, 10))
        for idx, im_neg in enumerate(im_negs):
            plt.subplot(6, 5, idx + 1)  # Adjust grid size based on the number of images
            plt.imshow(im_neg[0].squeeze(0).detach().numpy(), cmap='gray')
            plt.title(f"Step {idx + 1}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

        logging.info('Done')

    elif clevr and not our_model:
        config = load_config('/Users/carlahugod/Desktop/UNI/6sem/bach/energy-based-representation-learning/src/config/clevr_config.yml')
        dataset = Clevr(config, train=False)

        random_sampler = RandomSampler(dataset, replacement=True, num_samples=config.steps * config.batch_size) 
        train_dataloader = DataLoader(dataset, num_workers=config.data_workers, batch_size=config.batch_size, sampler=random_sampler, pin_memory=False)

        im,idx = next(iter(train_dataloader))
        


        # TODO: use dataloader, shape: batch size, 3, 64, 64
        
        state_dicts = torch.load('/Users/carlahugod/Desktop/UNI/6sem/bach/energy-based-representation-learning/src/models/clevr_our_code_steps_200.pth', torch.device('cpu'), weights_only=False)
        models = [LatentEBM128(config, 'clevr').to(torch.device('cpu')) for _ in range(len(state_dicts))]
        for i, model in enumerate(models):
            model.load_state_dict(state_dicts[i])
            model.eval()  # Set to evaluation mode

        latent = models[0].embed_latent(im)
        latents = torch.chunk(latent, 5, dim=1)

        tensor_img = im
        print(tensor_img.shape)
        # add specific batch size 12 to shape:
        # tensor_img = tensor_img.unsqueeze(0)
        # print(tensor_img.shape)

        latent = models[0].embed_latent(tensor_img)
        latents = torch.chunk(latent, config.components, dim=1)
        im_neg = torch.rand_like(tensor_img)
        im_negs = gen_image(latents, config, models, im_neg, 30)

        plt.imshow(tensor_img[0].detach().numpy().transpose(1,2,0))
        plt.title("Original Image")
        plt.axis("off")



        # Visualize each tensor in im_negs
        plt.figure(figsize=(15, 10))
        for idx, im_neg in enumerate(im_negs):
            plt.subplot(6, 5, idx + 1)  # Adjust grid size based on the number of images
            plt.imshow(im_neg[0].detach().numpy().transpose(1, 2, 0))  # Transpose to (H, W, C) for RGB images
            plt.title(f"Step {idx + 1}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

        logging.info('Done')



    elif not clevr and not our_model:
        print("Processing LatentEBM128 model")
        ckpt = torch.load('/Users/carlahugod/Desktop/UNI/6sem/bach/energy-based-representation-learning/comet/celebahq_128.pth', torch.device('cpu'))
        
        config = ckpt['FLAGS']
        state_dicts = ckpt['model_state_dict_0']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = LatentEBM128(config, 'celebahq_128').to(device)
        model.load_state_dict(state_dicts)
        models = [model for i in range(4)]

        tensor_img = dataset.__getitem__(0)[0].unsqueeze(0).expand(-1, 3, -1, -1)
        print(f"Input tensor shape: {tensor_img.shape}")

        latent = models[0].embed_latent(tensor_img)
        latents = torch.chunk(latent, config.components, dim=1)
        im_neg = torch.rand_like(tensor_img)
        im_negs = gen_image(latents, config, models, im_neg, 30)


        # Visualize each tensor in im_negs
        plt.figure(figsize=(15, 10))
        for idx, im_neg in enumerate(im_negs):
            plt.subplot(6, 5, idx + 1)  # Adjust grid size based on the number of images
            plt.imshow(im_neg[0].squeeze(0).detach().numpy().transpose(1,2,0), cmap='gray')
            plt.title(f"Step {idx + 1}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

        logging.info("LatentEBM128 processing complete")