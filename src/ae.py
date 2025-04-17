import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

# Define the Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # Output: 16 x 32 x 32
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Output: 32 x 16 x 16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: 64 x 8 x 8
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 32 x 16 x 16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 16 x 32 x 32
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 1 x 64 x 64
            nn.Sigmoid(),  # Normalize output to [0, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Define a Dataset Class
class NumpyDataset(Dataset):
    def __init__(self, data_path):
        self.data = []
        for file in sorted(os.listdir(data_path)):
            if file.endswith(".npy"):
                self.data.append(np.load(os.path.join(data_path, file)))
        self.data = np.concatenate(self.data, axis=0)  # Combine all .npy files
        self.data = self.data[:, None, :, :]  # Add channel dimension (1 for grayscale)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)

# Training Function
def train_autoencoder(data_path, batch_size=20, num_epochs=10, lr=0.001):
    # Load dataset
    dataset = NumpyDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss, and optimizer
    model = Autoencoder().cuda() if torch.cuda.is_available() else Autoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch in dataloader:
            batch = batch.cuda() if torch.cuda.is_available() else batch
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "autoencoder.pth")
    print("Model saved as autoencoder.pth")

    # Visualize some reconstructions
    visualize_reconstructions(model, dataloader)

# Visualization Function
def visualize_reconstructions(model, dataloader):
    model.eval()
    batch = next(iter(dataloader))
    batch = batch.cuda() if torch.cuda.is_available() else batch
    with torch.no_grad():
        reconstructions = model(batch)

    # Plot original and reconstructed images
    fig, axes = plt.subplots(2, 10, figsize=(15, 4))
    for i in range(10):
        # Original images
        axes[0, i].imshow(batch[i, 0].cpu().numpy(), cmap="gray")
        axes[0, i].axis("off")
        # Reconstructed images
        axes[1, i].imshow(reconstructions[i, 0].cpu().numpy(), cmap="gray")
        axes[1, i].axis("off")
    plt.suptitle("Top: Original Images, Bottom: Reconstructed Images")
    plt.show()

if __name__ == "__main__":
    # Path to the dataset
    data_path = "data"  # Update this path if necessary
    train_autoencoder(data_path, batch_size=20, num_epochs=10, lr=0.001)