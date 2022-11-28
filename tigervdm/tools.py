import numpy as np
import nibabel as nib
import SimpleITK as sitk
from UNet3d import UNet3d
from nilearn.image import resample_img

def download(url, file_name):
    import urllib.request
    import certifi
    import shutil
    import ssl
    context = ssl.create_default_context(cafile=certifi.where())
    #urllib.request.urlopen(url, cafile=certifi.where())
    with urllib.request.urlopen(url,
                                context=context) as response, open(file_name, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)

def get_b0_slice(ff):
    with open(ff) as f:
        bvals = f.readlines()[0].replace('\n', '').split(' ')
    return np.argmax(np.char.equal(bvals, '0'))

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