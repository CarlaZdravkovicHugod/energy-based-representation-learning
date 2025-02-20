import kagglehub
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from torch.utils.data import Dataset

class BrainDataset(Dataset):
    def __init__(self):
        self.path = kagglehub.dataset_download("balakrishcodes/brain-2d-mri-imgs-and-mask")
        self.data = None
    
    def get_data(self) -> None: return self.path
    
    def load_data(self, reload: bool = False) -> None:
        if not reload and self.data is not None: return
        self.data = []
        img_path = []
        for folder_name in os.listdir(self.path):
            path = os.path.join(self.path, folder_name)
            img_path += glob.glob(os.path.join(path, "*.h5"))
        for file in img_path:
            with h5py.File(file, "r") as f:
                X = np.array(f.get("x"))
                y = np.array(f.get("y"))
                y[y < 25] = 0.0
                y[y >= 25] = 1.0
                self.data.append((X, y))

    def visualize(self, idx: int) -> None:
        X, y = self.data[idx]
        plt.imshow(X)
        plt.show()

    def __len__(self) -> int: return len(self.data)
    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]: return self.data[idx]

if __name__ == "__main__":
    dataset = BrainDataset()
    print(dataset.get_data())
    dataset.load_data()
    print('Lenght of data set:', len(dataset))
    dataset.visualize(0)