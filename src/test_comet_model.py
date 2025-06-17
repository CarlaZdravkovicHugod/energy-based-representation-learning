import matplotlib.pyplot as plt
import torch
from comet_models import LatentEBM
from config.load_config import load_config
from dataloader import MRI2D, Clevr
from imageio.v2 import get_writer
import logging
from tqdm import tqdm
from easydict import EasyDict
import argparse

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


def main(checkpoint, index):
    # Load a comet model
    ckpt = torch.load(checkpoint, torch.device('cpu'))
    config = EasyDict(ckpt['FLAGS'])

    dataset = Clevr(config, train=False)
    state_dicts = ckpt['model_state_dict_0']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LatentEBM(config, 'clevr').to(device)
    model.load_state_dict(state_dicts)
    models = [model for i in range(config.components)]

    tensor_img = dataset.__getitem__(index)[0].unsqueeze(0).expand(-1, 3, -1, -1)
    print(f"Input tensor shape: {tensor_img.shape}")

    latent = models[0].embed_latent(tensor_img)
    latents = torch.chunk(latent, config.components, dim=1)
    im_neg = torch.rand_like(tensor_img)
    im_negs = gen_image(latents, config, models, im_neg, 30)

    im = tensor_img

    plt.imshow(im[0].detach().numpy().transpose(1,2,0))
    plt.title("Original Image")
    plt.axis("off")

    run_name = checkpoint.split("/")[-1].split(".")[0]
    gif_path = f"src/videos/clevr_{run_name}.gif"
    with get_writer(gif_path, mode="I", duration=0.13) as writer:
        for im_neg in im_negs:
            im_neg_np = im_neg[0].detach().cpu().numpy().transpose(1, 2, 0)
            writer.append_data((im_neg_np * 255).astype('uint8'))

    print(f"GIF saved at {gif_path}")

    print("Reconstructing images using specific latents...")
    plt.figure(figsize=(15, 10))
    for i, latent in enumerate(latents):
        im_neg = torch.rand_like(im)
        im_negs = gen_image(latents, config, models, im_neg, 60, idx=i)

        im_neg_np = im_negs[-1][0].detach().cpu().numpy().transpose(1, 2, 0)
        plt.subplot(1, len(latents), i + 1)
        plt.imshow(im_neg_np)
        plt.title(f"Component {i + 1}")
        plt.axis("off")
        plt.tight_layout()

    plt.savefig(f"src/videos/clevr_reconstructed_{run_name}.png")
    plt.show()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=False, default='src/models/clevr_comet_99900.pth.pth', help="Path to model checkpoint")
    parser.add_argument("--index", type=int, required=False, default=0, help="Index of image in dataset to reconstruct")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Using checkpoint: {args.checkpoint}")

    main(args.checkpoint, args.index)
    