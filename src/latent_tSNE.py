import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from ae import UNetAutoencoder, NumpyMRIDataset
from torch.utils.data import DataLoader
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import logging

# Initialize the UNetAutoencoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetAutoencoder(base_ch=256, skip_connections=False).to(device)

# Load the checkpoint
checkpoint_path = "src/models/autoencoder_best_model.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state'])
model.eval()

# Load a batch of images
dataset = NumpyMRIDataset(root="data/")
if len(dataset) == 0:
    raise ValueError("Dataset is empty. Please check the data path and ensure it contains images.")

dataloader = DataLoader(dataset, batch_size=20)
im = next(iter(dataloader))
images = im.to(device)
logging.info(f"Images shape: {images.shape}")

# Extract latent representations
logging.info("Encoding...")
with torch.no_grad():
    latents, _ = model.encode(images)

# Flatten the latent space for t-SNE
latents_flat = latents.view(latents.size(0), -1).cpu().numpy()
logging.info(f"Latents flattened shape: {latents_flat.shape}")
logging.info(f"Latents original shape: {latents.shape}")

# Perform t-SNE
perplex = min(30, latents_flat.shape[0] - 1)
logging.info(f"Perplexity: {perplex}")
tsne = TSNE(n_components=2, random_state=42, perplexity=perplex)  # Reduce to 2 dimensions for visualization
latents_tsne = tsne.fit_transform(latents_flat)

# Visualize the t-SNE results
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(latents_tsne[:, 0], latents_tsne[:, 1])
ax.set_title("t-SNE Visualization of Latent Space")
ax.set_xlabel("t-SNE Dimension 1")
ax.set_ylabel("t-SNE Dimension 2")

# Function to create an image annotation
def imscatter(x, y, images, ax=None, zoom=0.1):
    if ax is None:
        ax = plt.gca()
    for i in range(len(images)):
        img = images[i].squeeze().cpu().numpy()
        imagebox = OffsetImage(img, cmap="gray", zoom=zoom)
        ab = AnnotationBbox(imagebox, (x[i], y[i]), frameon=False)
        ax.add_artist(ab)

# Overlay images on the t-SNE plot
imscatter(latents_tsne[:, 0], latents_tsne[:, 1], images, ax=ax)
plt.tight_layout()
plt.show()

# Reconstruct images from the latent space
logging.info("Decoding...")
with torch.no_grad():
    reconstructed_images = model.decode(latents)

# Visualize original and reconstructed images
fig, axes = plt.subplots(2, len(images), figsize=(15, 5))
for i in range(len(images)):
    axes[0, i].imshow(images[i].squeeze().cpu().numpy(), cmap="gray")
    axes[0, i].axis("off")
    axes[1, i].imshow(reconstructed_images[i].squeeze().cpu().numpy(), cmap="gray")
    axes[1, i].axis("off")
plt.suptitle("Original (top) and Reconstructed (bottom) Images")
plt.tight_layout()
plt.show()