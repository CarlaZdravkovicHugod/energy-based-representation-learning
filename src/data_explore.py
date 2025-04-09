import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataloader import MRI2D
from config.load_config import load_config
import os
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Helvetica']})


config = load_config(os.path.abspath(os.path.join(os.path.dirname(__file__), 'config','2DMRI_config.yml')))
dataset = MRI2D(config)

org_img = np.load(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'T1w_sub-0001_x-5.npy')))
sample, _ = dataset[0]
sample2, _ = dataset[2]


# first top part of the image is always just a black line
# so remove it:
# sample_wo = sample[0][28:, 12:]  # remove the first row and first columns
# plt.imshow(sample_wo, cmap='gray')
# plt.title(f"Sample 1 without first 12 cols and 28 rows")
# plt.axis('off')
# plt.show()

# Visualize original image compared to sample
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(sample[0], cmap='gray')
plt.axis('off')
plt.title('Sample from Dataloader')
plt.subplot(1, 2, 2)
plt.imshow(org_img, cmap='gray')
plt.axis('off')
plt.title('Original Image')
plt.savefig(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'figures', 'org_vs_sample.png')), bbox_inches='tight')
plt.show()

# Visualize 2 data samples fra dataloader
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(sample[0], cmap='gray')
plt.title(f"Random sample 1")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(sample2[0], cmap='gray')
plt.title(f"Random sample 2")
plt.axis('off')
plt.savefig(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'figures', 'random_samples.png')), bbox_inches='tight')
plt.show()

# Visualize the distribution of the data
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(sample[0].flatten(), bins=50, color='blue', alpha=0.7)
plt.title(f"Distribution of pixel values in random sample")
plt.xlabel("Pixel value")
plt.ylabel("Frequency")
plt.grid()
plt.subplot(1, 2, 2)
plt.hist(org_img.flatten(), bins=50, color='blue', alpha=0.7)
plt.title(f"Distribution of pixel values in original image")
plt.xlabel("Pixel value")
plt.ylabel("Frequency")
plt.grid()
plt.savefig(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'figures', 'histograms_comparison.png')))
plt.show()


# Correlation matrix
# corr_matrix = torch.corrcoef(sample[0].T)
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=False, cmap='grey', vmin=sample[0].min(), vmax=sample[0].max(), square=False)
# plt.title(f"Correlation matrix of sample {idx}")
# plt.show()

