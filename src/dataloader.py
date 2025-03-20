import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from pathlib import Path
import seaborn as sns
import logging
import torch.utils.data as data
from src.config.load_config import Config
import torch
from glob import glob
from skimage.transform import resize as imresize
from imageio import imread
from src.config.load_config import load_config
from torch.nn.functional import normalize

class BrainDataset(Dataset):
    def __init__(self, config: Config):
        """
        Initialize the BrainDataset with the path to the data directory.
        """
        self.path = Path(__file__).absolute().parent.parent / 'data'
        self.data = None
        self.config = config
        self.test_run = True

    def get_data(self) -> str: return self.path

    def load_data(self, reload: bool = False) -> None:
        if not reload and self.data is not None: return
        self.data = []
        for filename in os.listdir(self.path):
            if self.test_run and len(self.data) >= 2: break
            self.data.append(np.load(str(self.path / filename)))

        logging.info(f'Data loaded: Number of datapoints = {len(self.data)}')
        assert len(self.data) == 2 or not self.test_run, f'Expected 2 datapoints, got {len(self.data)}'
        return self.data

    def visualize(self, idx: int) -> None:
        if self.data is None:
            print("Data is not loaded.")
            return
        if idx < 0 or idx >= len(self.data):
            print("Index out of bounds.")
            return
        X = self.data[idx]
        sns.heatmap(X)
        plt.show()

    def __len__(self) -> int:
        if self.data is None:
            return 0
        return len(self.data)
    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]: return self.data[idx]


    # Define a function that reads the data from the path and returns the tensors with 3 dimensions as 3,64,64 the same way as Clevr


class Clevr(data.Dataset):
    def __init__(self, config: Config, stage=0):
        self.path = str(Path(__file__).absolute().parent.parent / Path(config.data_path) / "*.png")
        self.images = sorted(glob(self.path))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        im_path = self.images[index]
        im = imread(im_path)
        im = imresize(im, (64, 64))[:, :, :3]

        im = torch.Tensor(im).permute(2, 0, 1)

        return im, index
    
class MRI2D(data.Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): Directory with all the `.npy` files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        data_dir = '/Users/carlahugod/Desktop/UNI/6sem/bach/energy-based-representation-learning/data'
        self.transform = transform
        self.files = sorted(glob(os.path.join(data_dir, '*.npy')))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if isinstance(idx, (list, np.ndarray)):
            idx = idx.tolist()

        npy_path = self.files[idx]
        sample = np.load(npy_path)
        # TODO: figure out how the data looks before anything is done to it and compare to clever data

        nonzero_mask = sample > 0  

        # Normalize only nonzero values to range [0, 1]
        img_min = np.min(sample[nonzero_mask]) 
        img_max = np.max(sample[nonzero_mask])    
        img_norm = np.zeros_like(sample)  
        img_norm[nonzero_mask] = (sample[nonzero_mask] - img_min) / (img_max - img_min)


        if self.transform:
            sample = self.transform(img_norm)

        # Convert to torch tensor and resize to (3, x, x)
        sample = torch.Tensor(img_norm)
        if sample.dim() == 2:  # If the sample is 2D, expand to 3D
            sample = sample.unsqueeze(0).repeat(3, 1, 1)
        elif sample.size(0) == 1:  # If the sample has a single channel, repeat it to have 3 channels
            sample = sample.repeat(3, 1, 1)


        # Reduce sample to size torch.size(3, 64, 64)
        # sample = sample.view(3, 64, 64)

        # TODO: use bicubar or adaptive pooling and revert changes in train
        # 1: bicubar
        import torch.nn.functional as F
        x_resized = F.interpolate(sample.unsqueeze(0), size=(64, 64), mode='bicubic', align_corners=False)
        sample = x_resized.squeeze(0)

        # 2: Adaptive pooling:
        # import torch.nn as nn
        # pool = nn.AdaptiveAvgPool2d((64, 64))  # Averaging retains more meaningful features
        # sample = pool(sample)  # Output: [3, 64, 64]
        
        
        #print(f'From dataset.py, using dataset MRI2D, sample: {sample.shape}, index: {idx}')

        # TODO: Some kidn of downsampling, Normalize, or maybe subtract mean and divide by std
        # normalized_sample = normalize(sample)

        
        return sample, idx


if __name__ == "__main__":
    # dataset = BrainDataset('src/config/test.yml')
    # print(dataset.get_data())
    # data1 = dataset.load_data()
    # print('Length of data set:', len(dataset))
    #dataset.visualize(0)

    config = load_config("src/config/clevr_config.yml") 
    # clebr jas 11407 files
    # these number in the tensor or very small 0.7 ish
    print(config)
    d2 = Clevr(config)
    print(len(d2))


    d3 = MRI2D('/Users/carlahugod/Desktop/UNI/6sem/bach/energy-based-representation-learning/data/*.npy')
    item = d3.__getitem__(0)
    print(item[0].shape)
    print(len(d3))
    # this has 2820 files
    # all the values in the tensors are really large, ranges from -90 to 3000
