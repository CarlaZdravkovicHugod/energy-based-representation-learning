import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from ae import UNetAutoencoder, NumpyMRIDataset
from torch.utils.data import DataLoader
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

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
print(f"Images shape: {images.shape}")

# Extract latent representations
with torch.no_grad():
    latents, _ = model.encode(images)

# Flatten the latent space for PCA
latents_flat = latents.view(latents.size(0), -1).cpu().numpy()
print(f"Latents flattened shape: {latents_flat.shape}")
print(f"Latents original shape: {latents.shape}")

# Perform PCA
pca = PCA(n_components=latents_flat[0])  # Keep only the first 3 principal components
latents_pca = pca.fit_transform(latents_flat)

# Visualize the PCA results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot PC1 vs PC2
axes[0].scatter(latents_pca[:, 0], latents_pca[:, 1])
axes[0].set_xlabel("Principal Component 1")
axes[0].set_ylabel("Principal Component 2")
axes[0].set_title("PC1 vs PC2")

# Plot PC1 vs PC3
axes[1].scatter(latents_pca[:, 0], latents_pca[:, 2])
axes[1].set_xlabel("Principal Component 1")
axes[1].set_ylabel("Principal Component 3")
axes[1].set_title("PC1 vs PC3")

# Plot PC2 vs PC3
axes[2].scatter(latents_pca[:, 1], latents_pca[:, 2])
axes[2].set_xlabel("Principal Component 2")
axes[2].set_ylabel("Principal Component 3")
axes[2].set_title("PC2 vs PC3")

plt.tight_layout()
plt.show()

# Function to create an image annotation
def imscatter(x, y, images, ax=None, zoom=0.1):
    if ax is None:
        ax = plt.gca()
    for i in range(len(images)):
        img = images[i].squeeze().cpu().numpy()
        imagebox = OffsetImage(img, cmap="gray", zoom=zoom)
        ab = AnnotationBbox(imagebox, (x[i], y[i]), frameon=False)
        ax.add_artist(ab)

# Create a new figure for the plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot PC1 vs PC2 with images
axes[0].scatter(latents_pca[:, 0], latents_pca[:, 1], alpha=0)
imscatter(latents_pca[:, 0], latents_pca[:, 1], images, ax=axes[0])
axes[0].set_title("PC1 vs PC2")
axes[0].set_xlabel("Principal Component 1")
axes[0].set_ylabel("Principal Component 2")

# Plot PC1 vs PC3 with images
axes[1].scatter(latents_pca[:, 0], latents_pca[:, 2], alpha=0)
imscatter(latents_pca[:, 0], latents_pca[:, 2], images, ax=axes[1])
axes[1].set_title("PC1 vs PC3")
axes[1].set_xlabel("Principal Component 1")
axes[1].set_ylabel("Principal Component 3")

# Plot PC2 vs PC3 with images
axes[2].scatter(latents_pca[:, 1], latents_pca[:, 2], alpha=0)
imscatter(latents_pca[:, 1], latents_pca[:, 2], images, ax=axes[2])
axes[2].set_title("PC2 vs PC3")
axes[2].set_xlabel("Principal Component 2")
axes[2].set_ylabel("Principal Component 3")

plt.tight_layout()
plt.show()

# Reconstruct images from the latent space
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

# Calculate cumulative variance explained
cumulative_variance = pca.explained_variance_ratio_.cumsum()

# Plot cumulative variance explained
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Variance Explained')
plt.title('Cumulative Variance Explained by PCA Components')
plt.xticks(range(1, len(cumulative_variance) + 1))
plt.grid()
plt.show()