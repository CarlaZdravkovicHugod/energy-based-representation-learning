import kagglehub
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import nibabel as nib
import nilearn.image as nim
import glob
import scipy.ndimage as ndi
from PIL import Image
from pathlib import Path
import seaborn as sns

class BrainDataset(Dataset):
    def __init__(self):
        """
        Initialize the BrainDataset with the path to the data directory.
        """
        self.path = Path(__file__).absolute().parent.parent / 'data'
        self.data = None
    
    def get_data(self) -> str: return self.path
    
    def load_data(self, reload: bool = False) -> None:
        if not reload and self.data is not None: return
        self.data = []
        for filename in os.listdir(self.path):
            self.data.append(np.load(str(self.path / filename)))
        
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

if __name__ == "__main__":
    dataset = BrainDataset()
    print(dataset.get_data())
    dataset.load_data()
    print('Length of data set:', len(dataset))
    dataset.visualize(0)