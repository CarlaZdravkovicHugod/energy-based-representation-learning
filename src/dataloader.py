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
import torch.nn.functional as F
import sklearn.preprocessing as skp

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

    def __len__(self) -> int: return 0 if self.data is None else len(self.data)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        assert self.data is not None, "Data is not loaded"
        rand_idx = np.random.randint(0, len(self.data))
        return self.data[rand_idx]


    # Define a function that reads the data from the path and returns the tensors with 3 dimensions as 3,64,64 the same way as Clevr


class Clevr(data.Dataset):
    def __init__(self, config: Config, train: bool):
        self.path = str(Path(__file__).absolute().parent.parent / Path(config.data_path) / "*.png")
        self.all_images = sorted(glob(self.path))
        self.images = self.all_images[:int(len(self.all_images) * 0.8)] if train else self.all_images[int(len(self.all_images) * 0.8):]
        self.train = train

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
    
        sample = (sample - sample.min()) / (sample.max() - sample.min())
        sample = F.pad(torch.tensor(sample), (0, 0, 256-sample.shape[0], 0))
        sample = sample.unsqueeze(0) # add channel dimension explicitly
        
        return sample, idx


if __name__ == "__main__":
    d3 = MRI2D('/Users/carlahugod/Desktop/UNI/6sem/bach/energy-based-representation-learning/data/*.npy')
    item = d3.__getitem__(0)
    print(item[0].shape)
    print(len(d3))
    # this has 2820 files
    # all the values in the tensors are really large, ranges from -90 to 3000
