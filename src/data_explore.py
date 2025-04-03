import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from dataloader import MRI2D
from config.load_config import load_config
import torch

# 1. Visualize the data
# 2. Visualize the distribution of the data
# 3. Histogram if relevant
# 4. Correlation matrix within an image


config = load_config('/Users/carlahugod/Desktop/UNI/6sem/bach/energy-based-representation-learning/src/config/2DMRI_config.yml')
dataset = MRI2D(config)

# first top part of the image is always just a black line
# so remove it:
sample, idx = dataset[1]

sample2 = sample[0][28:, 12:]  # remove the first row and first columns
plt.imshow(sample2, cmap='gray')
plt.title(f"Sample {idx} without first 12 cols and 28 rows")
plt.axis('off')
plt.show()

# Visualize the data
plt.imshow(sample[0], cmap='gray')
plt.title(f"Sample {idx}")
plt.axis('off')
plt.show()


# Visualize the distribution of the data
plt.hist(sample[0].flatten(), bins=50, color='blue', alpha=0.7)
plt.title(f"Distribution of pixel values in sample {idx}")
plt.xlabel("Pixel value")
plt.ylabel("Frequency")
plt.show()


# Correlation matrix
corr_matrix = torch.corrcoef(sample[0].T)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=False, cmap='grey', vmin=sample[0].min(), vmax=sample[0].max(), square=False)
plt.title(f"Correlation matrix of sample {idx}")
plt.show()

