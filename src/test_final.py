import matplotlib.pyplot as plt
import torch
from comet_models import LatentEBM, LatentEBM128
from config.load_config import load_config
from dataloader import MRI2D, Clevr
from imageio.v2 import get_writer
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler
import argparse

# Train AE to reconstruct


def reconstruct_and_plot(config_path, checkpoint_path, num_steps, batch_size):
    """
    Test a model on a given dataset and visualize the results.

    Args:
        config_path (str): Path to the configuration file.
        checkpoint_path (str): Path to the model checkpoint.
        num_steps (int): Number of steps for image generation.
        batch_size (int): Batch size for testing.
    """
    # Load configuration
    config = load_config(config_path)
    logging.info(f"Loaded config: {config}")

    run_name = checkpoint_path.split("/")[-1].split(".")[0]
    logging.info(f"Run name: {run_name}")

    dataset_type = config.dataset
    if dataset_type == "2DMRI":
        dataset = MRI2D(config)
    elif dataset_type == "clevr":
        dataset = Clevr(config, train=False)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model_type = config.model
    if model_type == "LatentEBM":
        model_class = LatentEBM
    elif model_type == "LatentEBM128":
        model_class = LatentEBM128
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dicts = checkpoint if isinstance(checkpoint, list) else [checkpoint]
    models = [model_class(config, dataset_type).to(device) for _ in range(len(state_dicts))]
    for i, model in enumerate(models):
        model.load_state_dict(state_dicts[i])
        model.eval()  # Set to evaluation mode

    # Prepare data
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=config.data_workers)
    im, idx = next(iter(dataloader))
    im = im.to(device)

    # Embed latent
    latent = models[0].embed_latent(im)
    latents = torch.chunk(latent, config.components, dim=1)
    # each latent is a component, and has size (batch_size, latent_dim)

    im_neg = torch.rand_like(im)
    im_negs = gen_image(latents, config, models, im_neg, num_steps)

    plt.imshow(im[0].detach().numpy().transpose(1,2,0))
    plt.title("Original Image")
    plt.axis("off")

    gif_path = f"src/videos/{dataset_type}_{run_name}.gif"
    with get_writer(gif_path, mode="I", duration=0.13) as writer:  # `duration` sets the delay between frames in seconds
        for im_neg in im_negs:
            im_neg_np = im_neg[0].detach().cpu().numpy().transpose(1, 2, 0)
            im_neg_np = im_neg_np.repeat(3, axis=-1)  # Convert all images to RGB
            writer.append_data((im_neg_np * 255).astype('uint8'))

    print(f"GIF saved at {gif_path}")

    # TODO: plot the im_neg as heay surface

    print("Reconstructing images using specific latents...")
    plt.figure(figsize=(15, 10))
    for i, latent in enumerate(latents):
        im_neg = torch.rand_like(im)  # Initialize a random image
        im_negs = gen_image(latents, config, models, im_neg, num_steps*2, idx=i)  # Use only the i-th component

        # Plot the final generated image for this component
        im_neg_np = im_negs[-1][0].detach().cpu().numpy().transpose(1, 2, 0)
        plt.subplot(1, len(latents), i + 1)
        plt.imshow(im_neg_np)
        plt.title(f"Component {i + 1}")
        plt.axis("off")
        plt.tight_layout()

    # save image:
    plt.savefig(f"src/videos/{dataset_type}_reconstructed_{run_name}.png")
    plt.show()


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, help="Path to config file", default='src/config/2DMRI_config.yml')
    parser.add_argument("--checkpoint", type=str, required=False, help="Path to model checkpoint", default='src/models/models_UN652_10100.pth')
    args = parser.parse_args()

    run_name = args.checkpoint.split("/")[-1].split(".")[0]
    logging.info(f"Run name: {run_name}")

    reconstruct_and_plot(
        args.config,
        args.checkpoint,
        num_steps=60,
        batch_size=12
    )