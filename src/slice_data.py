import os
import numpy as np
import nibabel as nib
import glob
import scipy.ndimage as ndi
from PIL import Image


# Scratch folders for HPC:
#  $ ls -ld /work3/s224195 /work3/s224209
# drwxr-x--- 2 s224195 s224209s224195 0 Feb 19 15:05 /work3/s224195
# drwxr-x--- 2 s224209 s224209s224195 0 Feb 19 15:05 /work3/s224209


def get_slice_data():
        """
        This function is used to extract the slice data from the 3D MRI images.
        The function will extract the slices at x = -5 and x = 5 from the center of the images.
        The extracted slices will be saved as binary numpy arrays.
        """

    # this should be the hpc path to data when it is uploaded
        fns = glob.glob('/mnt/projects/KHM/nobackup/SRPBS/SRPBS_OPEN/out*/fmriprep/sub-*/anat/sub-*_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz')
        outdir = os.path.join(os.path.dirname(__file__), 'slicedata/')


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

if __name__ == "__main__":
    get_slice_data()
    print('Slices extracted and saved as binary numpy arrays')
