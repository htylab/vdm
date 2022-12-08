import sys
import argparse
import glob
import os
import numpy as np
import nibabel as nib
import tqdm
import torch
from UNet3d import UNet3d
from scipy.ndimage import gaussian_filter
from lib_tool import *
from lib_vdm import *
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('input',  type=str, nargs='+', help='Path to the input image, can be a folder for the specific format(nii.gz)')
parser.add_argument('-o', '--output', default=None, help='File path for output image, default: the directory of input files')
parser.add_argument('-b0', '--b0_index', default=None, type=str, help='The index of b0 slice or the .bval file, default: 0 (the first slice)')
parser.add_argument('-n', '--no_resample', action='store_true', help='Don\'t resample to 1.7x1.7x1.7mm3')
parser.add_argument('-m', '--dmap', action='store_true', help='Producing the virtual displacement map')
parser.add_argument('-g', '--gpu', action='store_true', help='Using GPU')

args = parser.parse_args() 

ffs = args.input
if os.path.isdir(args.input[0]):
    ffs = glob.glob(os.path.join(args.input[0], '*.nii'))
    ffs += glob.glob(os.path.join(args.input[0], '*.nii.gz'))

elif '*' in args.input[0]:
    ffs = glob.glob(args.input[0])

output_dir = args.output

device = 'cuda' if (args.gpu and torch.cuda.is_available()) else 'cpu'

if args.b0_index is None:
    b0_index = 0
elif os.path.exists(args.b0_index.replace('.bval', '') + '.bval'):
    b0_index = get_b0_slice(args.b0_index.replace('.bval', '') + '.bval')
else:
    b0_index = int(args.b0_index)
    
resample = (not args.no_resample)

model_path = './models'
os.makedirs(model_path, exist_ok=True)
model_file = os.path.join(model_path,'vdm_model_v1.pt')
model_url = 'https://github.com/htylab/vdm/releases/download/modelhub/vdm_model_v1.pt'
if not os.path.exists(model_file):
    print(f'Downloading model files....')
    print(model_url, model_file)
    download(model_url, model_file)
    print('Download finished...')

    

NET = UNet3d(in_channels=1, n_classes=1, n_channels=24, z_pooling=True).to(device)
NET.load_state_dict(torch.load(model_file))
    
for f in tqdm.tqdm(ffs):
    
    temp = nib.load(f)
    affine = temp.affine
    zoom = temp.header.get_zooms()[0:3]
    vol_org = temp.get_fdata()
    if len(temp.shape)>3:
        vol = vol_org.copy()[...,b0_index]
    else:
        vol = vol_org.copy()
        
    vol[vol<0] = 0
    
    if resample:
        resample_nii = resample_to_new_resolution(nib.Nifti1Image(vol, affine), target_resolution=(1.7, 1.7, 1.7), target_shape=None, interpolation='continuous')
        vol_resize = resample_nii.get_fdata()
        vol_resize = vol_resize / np.max(vol_resize)
    else:
        vol_resize = vol / np.max(vol)
    
    vol_d = torch.from_numpy(vol_resize).to(device).float()   
    logits = NET(vol_d[None, ...][None, ...])

    df_map_org = logits[0,0, ...].cpu().detach().numpy()
    
    
    if resample:
        df_map = resample_to_new_resolution(nib.Nifti1Image(df_map_org, resample_nii.affine), target_resolution=zoom, target_shape=vol.shape, interpolation='linear').get_fdata() / 1.7 * zoom[1]
    else:
        df_map = df_map_org
    
    df_map_f = np.array(df_map*0, dtype='float64')
    for nslice in np.arange(vol.shape[2]):
        df_map_slice = gaussian_filter(df_map[..., nslice], sigma=1.5).astype('float64')
        df_map_f[..., nslice] = df_map_slice

    vol_out = vol_org*0
    if len(vol_org.shape)==4:
        for bslice in range(vol_org.shape[3]):
            vol_out[...,bslice] = apply_vdm_3d(vol_org[...,bslice], df_map_f, AP_RL='AP')
    else:
        vol_out = apply_vdm_3d(vol_org, df_map_f, AP_RL='AP')

    
    if output_dir is None:
        result_dir = os.path.dirname(os.path.abspath(f))
    else:
        result_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    result = nib.Nifti1Image(vol_out.astype(temp.get_data_dtype()), affine)
    fn = os.path.basename(f).replace('.nii.gz', '_vdmi.nii.gz')
    nib.save(result, os.path.join(result_dir, fn))
    
    if args.dmap:
        result = nib.Nifti1Image(df_map_f, affine)
        result.header.set_zooms(zoom)
        fn = os.path.basename(f).replace('.nii.gz', '_vdm.nii.gz')
        nib.save(result, os.path.join(result_dir, fn))