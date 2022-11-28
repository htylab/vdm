import sys
import argparse
import glob
import os
from os.path import basename, join
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
import tqdm
import warnings
from UNet3d import UNet3d
warnings.filterwarnings("ignore")

from scipy.ndimage import gaussian_filter
from nilearn.image import resample_img

parser = argparse.ArgumentParser()
parser.add_argument('input',  type=str, nargs='+', help='Path to the input image, can be a folder for the specific format(nii.gz)')
parser.add_argument('output', help='File path for output segmentation.')
parser.add_argument('-g', '--gpu', action='store_true', help='Using GPU')

args = parser.parse_args() 

ffs = args.input
if os.path.isdir(args.input[0]):
    ffs = glob.glob(os.path.join(args.input[0], '*.nii'))
    ffs += glob.glob(os.path.join(args.input[0], '*.nii.gz'))

elif '*' in args.input[0]:
    ffs = glob.glob(args.input[0])

result_dir = args.output
os.makedirs(result_dir, exist_ok=True)

device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'

model_path = 'VDM_model_v1.pt'


    
def resample_to_new_resolution(data_nii, target_resolution, target_shape=None, interpolation='continuous'):
    affine = data_nii.affine
    target_affine = affine.copy()
    factor = np.zeros(3)
    for i in range(3):
        factor[i] = target_resolution[i] / np.sqrt(affine[0, i]**2 + affine[1, i]**2 + affine[2, i]**2)
        target_affine[:3, i] = target_affine[:3, i]*factor[i]
        
    new_nii = resample_img(data_nii, target_affine=target_affine, target_shape=target_shape, interpolation=interpolation)
    return new_nii
    
def apply_vdm_3d(ima, vdm, readout=1,  AP_RL='AP'):
    
    if AP_RL=='AP':
        arr = np.stack([vdm*0, vdm*readout, vdm*0], axis=-1)
    else:
        arr = np.stack([vdm*0, vdm*0, vdm*readout], axis=-1)
    displacement_image = sitk.GetImageFromArray(arr, isVector=True)
    
    jac = sitk.DisplacementFieldJacobianDeterminant(displacement_image)
    tx = sitk.DisplacementFieldTransform(displacement_image)
    ref = sitk.GetImageFromArray(ima*0)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref)
    resampler.SetInterpolator(sitk.sitkLinear) #sitkNearestNeighbor, sitk.sitkLinear
    resampler.SetTransform(tx)

    new_ima = resampler.Execute(sitk.GetImageFromArray(ima))
    new_ima = sitk.GetArrayFromImage(new_ima)
    jac_np = sitk.GetArrayFromImage(jac)
    return new_ima*jac_np

NET = UNet3d(in_channels=1, n_classes=1, n_channels=24, z_pooling=True).to(device)
NET.load_state_dict(torch.load(model_path))
    
for f in tqdm.tqdm(ffs):
    
    temp = nib.load(f)
    affine = temp.affine
    zoom = temp.header.get_zooms()[0:3]
    vol_org = temp.get_fdata()
    if len(temp.shape)>3:
        vol = vol_org.copy()[...,0]
    else:
        vol = vol_org.copy()

        
    vol[vol<0] = 0
    
    resample_nii = resample_to_new_resolution(nib.Nifti1Image(vol, affine), target_resolution=(1.75, 1.75, 1.75), target_shape=None, interpolation='continuous')
    vol_resize = resample_nii.get_fdata()
    vol_resize = vol_resize / np.max(vol_resize)
    
    vol_d = torch.from_numpy(vol_resize).to(device).float()   
    logits = NET(vol_d[None, ...][None, ...])

    df_map_org = logits[0,0, ...].cpu().detach().numpy()
    
    
    df_map = resample_to_new_resolution(nib.Nifti1Image(df_map_org, resample_nii.affine), target_resolution=zoom, target_shape=vol.shape, interpolation='linear').get_fdata() / 1.75 * zoom[1]
    
    df_map_f = np.array(df_map*0, dtype='float64')
    for nslice in np.arange(vol.shape[2]):
        df_map_slice = gaussian_filter(df_map[..., nslice], sigma=1.5).astype('float64')
        df_map_f[..., nslice] = df_map_slice

    vol_out = vol_org*0
    for bslice in range(vol_org.shape[3] if len(vol_org.shape)==4 else 1):
        vol_out[...,bslice] = apply_vdm_3d(vol_org[...,bslice], df_map_f, AP_RL='AP')

    
    
    result = nib.Nifti1Image(vol_out.astype(temp.get_data_dtype()), affine)
    fn = basename(f).replace('.nii.gz', '_vdmi.nii.gz')
    nib.save(result, join(result_dir, fn))
    
    result = nib.Nifti1Image(df_map, affine)
    result.header.set_zooms(zoom)
    fn = basename(f).replace('.nii.gz', '_vdm.nii.gz')
    nib.save(result, join(result_dir, fn))