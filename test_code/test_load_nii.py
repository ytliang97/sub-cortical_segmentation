import nibabel as nib


folder = '../data/MICCAI_MultiAtlasChallenge2012_corrected/Training/'

name = '1000_3.nii'

nib.Nifti1Header.quaternion_threshold = -1e-06

path = folder + name

print(path)
print(type(path))

file = nib.load(path)

data = file.get_data()