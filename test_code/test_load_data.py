import os

import numpy as np
from nibabel import load as load_nii

import nibabel as nib

name = '1000_3.nii.gz'
label_name = '1000_3_glm.nii.gz'
dir_name = '../data/MICCAI2012_train_samples/'

nib.Nifti1Header.quaternion_threshold = -1e-06


subjects = [f for f in sorted(os.listdir(dir_name)) if os.path.isdir(os.path.join(dir_name, f))]
print(subjects)
image_names = [os.path.join(dir_name, subject, name) for subject in subjects]
print(image_names)
images = [load_nii(name).get_data() for name in image_names] 
images_norm = [(im.astype(np.float32) - im[np.nonzero(im)].mean()) / im[np.nonzero(im)].std() for im in images]
    
# load labels 
label_names = [os.path.join(dir_name, subject, label_name) for subject in subjects]
print(label_names)
labels = [load_nii(name).get_data() for name in label_names]

