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


    def get_slice_data():
        """
        This function is used to extract the slice data from the 3D MRI images.
        The function will extract the slices at x = -5 and x = 5 from the center of the images.
        The extracted slices will be saved as binary numpy arrays.
        """

    # this should be the hpc path:
        fns = glob.glob('/mnt/projects/KHM/nobackup/SRPBS/SRPBS_OPEN/out*/fmriprep/sub-*/anat/sub-*_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz')
        
        # make automatic:
        outdir = '/mnt/projects/KHM/nobackup/SRPBS/SRPBS_OPEN/slicedata/'


        y = np.arange(-128, 100, 1)
        z = np.arange(-90, 108, 1)

        midslicel = np.array(np.meshgrid(-5, y, z, indexing='ij'))
        midslicer = np.array(np.meshgrid(5, y, z, indexing='ij'))

        for fn in fns:
            vol = nib.load(fn)
            iM = np.linalg.inv(vol.affine)
        
            vcoords = iM[:3,:3]@midslicel.reshape(3,-1) + iM[:3,3,None]
            data = ndi.map_coordinates(vol.dataobj, vcoords, order=5)
            data = data.reshape(y.shape[0], z.shape[0])
            np.save(outdir+'T1w_'+fn.split('/')[-3]+'_x-5.npy', data)

        
            vcoords = iM[:3,:3]@midslicer.reshape(3,-1) + iM[:3,3,None]
            data = ndi.map_coordinates(vol.dataobj, vcoords, order=5)
            data = data.reshape(y.shape[0], z.shape[0])
        
            np.save(outdir+'T1w_'+fn.split('/')[-3]+'_x5.npy', data)
        
        return data


    def __len__(self) -> int: return len(self.data)
    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]: return self.data[idx]

if __name__ == "__main__":
    dataset = BrainDataset()
    print(dataset.get_data())
    print(dataset.get_slice_data())
    dataset.load_data()
    print('Length of data set:', len(dataset))
    dataset.visualize(0)