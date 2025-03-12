import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from pathlib import Path
import seaborn as sns
import logging

from src.config.load_config import Config

class BrainDataset(Dataset):
    def __init__(self, config: Config):
        """
        Initialize the BrainDataset with the path to the data directory.
        """
        self.path = Path(__file__).absolute().parent.parent / 'data'
        self.data = None
        self.config = config
        self.test_run = config.test_run

    def get_data(self) -> str: return self.path

    def load_data(self, reload: bool = False) -> None:
        if not reload and self.data is not None: return
        self.data = []
        for filename in os.listdir(self.path):
            if self.test_run and len(self.data) >= 2: break
            self.data.append(np.load(str(self.path / filename)))

        logging.info(f'Data loaded: Number of datapoints = {len(self.data)}')
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