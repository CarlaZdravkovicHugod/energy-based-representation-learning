import matplotlib.pyplot as plt
import torch
from comet_models import LatentEBM
from config.load_config import load_config
from dataloader import MRI2D, Clevr
import matplotlib.pyplot as plt
from imageio.v2 import get_writer
import logging
from tqdm import tqdm
from easydict import EasyDict

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

    # Load a good comet model
    ckpt = torch.load('/Users/carlahugod/Desktop/UNI/6sem/bach/energy-based-representation-learning/src/models/clevr_comet_model_400.pth', torch.device('cpu'))
    config = EasyDict(ckpt['FLAGS'])

    dataset = Clevr(config, train=False)
    state_dicts = ckpt['model_state_dict_0']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LatentEBM(config, 'celebahq_128').to(device)
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


