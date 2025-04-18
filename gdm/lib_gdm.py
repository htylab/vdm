from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt2d
from os.path import basename, join
import SimpleITK as sitk
from os.path import join, basename, isdir
import numpy as np
import nibabel as nib
# import onnxruntime as ort
from scipy.special import softmax
from nilearn.image import resample_img


nib.Nifti1Header.quaternion_threshold = -100
def get_mode(model_ff):
    seg_mode, version, model_str = basename(model_ff).split('_')[1:4]  # aseg43, bet

    #print(seg_mode, version , model_str)

    return seg_mode, version, model_str

def run(model_ff, input_data, b0_index, GPU, resample=True):
    import onnxruntime as ort
    
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 4

    if GPU and (ort.get_device() == "GPU"):
        #ort.InferenceSession(model_file, providers=['CPUExecutionProvider'])
        session = ort.InferenceSession(model_ff,
                                       providers=['CUDAExecutionProvider'],
                                       sess_options=so)
    else:
        session = ort.InferenceSession(model_ff,
                                       providers=['CPUExecutionProvider'],
                                       sess_options=so)

    gdm_mode, _, model_type = get_mode(model_ff) #gdmmode: 3dunet, gan    
    AP_RL = 'RL' if 'hcp' in model_type else 'AP'

    orig_data = input_data  
    
    gdm_pred = gernerate_gdm(gdm_mode, session, orig_data, b0_index, resample=resample)

    output_vol = np.zeros(orig_data.shape)
    orig_data3d = orig_data.get_fdata()
    if len(orig_data.shape)==4:
        
        for bslice in range(orig_data.shape[3]):
            output_vol[...,bslice] = apply_gdm_3d(orig_data3d[...,bslice], gdm_pred, AP_RL=AP_RL)
            
    else:
        
        output_vol = apply_gdm_3d(orig_data3d, gdm_pred, AP_RL=AP_RL)

    

    return output_vol, gdm_pred


def read_file(model_ff, input_file):
    return nib.load(input_file)


def write_file(model_ff, input_file, output_dir, vol_out, inmem=False, postfix='gdmi'):

    if not isdir(output_dir):
        print('Output dir does not exist.')
        return 0

    output_file = basename(input_file).replace('.nii.gz', '').replace('.nii', '') 
    output_file = output_file + f'_{postfix}.nii.gz'
    output_file = join(output_dir, output_file)
    print('Writing output file: ', output_file)

    input_nib = nib.load(input_file)
    affine = input_nib.affine
    zoom = input_nib.header.get_zooms()
    

    if postfix=='gdm':
        result = nib.Nifti1Image(vol_out, affine)
    else:
        result = nib.Nifti1Image(vol_out.astype(input_nib.get_data_dtype()), affine)
        result.header.set_zooms(zoom)


    if not inmem:
        nib.save(result, output_file)

    return output_file, result


def predict(model, data):
    import onnxruntime as ort
    if model.get_inputs()[0].type == 'tensor(float)':
        return model.run(None, {model.get_inputs()[0].name: data.astype('float32')}, )[0]
    else:
        return model.run(None, {model.get_inputs()[0].name: data.astype('float64')}, )[0]

    
def get_b0_slice(ff):
    with open(ff) as f:
        bvals = f.readlines()[0].replace('\n', '').split(' ')
    bvals = [int(bval) for bval in bvals]
    return np.argmin(bvals)

    
def resample_to_new_resolution(data_nii, target_resolution, target_shape=None, interpolation='continuous'):
    affine = data_nii.affine
    target_affine = affine.copy()
    factor = np.zeros(3)
    for i in range(3):
        factor[i] = target_resolution[i] / np.sqrt(affine[0, i]**2 + affine[1, i]**2 + affine[2, i]**2)
        target_affine[:3, i] = target_affine[:3, i]*factor[i]
        
    new_nii = resample_img(data_nii, target_affine=target_affine, target_shape=target_shape, interpolation=interpolation)
    return new_nii


def apply_gdm_2d(ima, gdm, readout=1, AP_RL='AP'):

    if AP_RL == 'AP':
        arr = np.stack([gdm*readout, gdm*0], axis=-1)
    else:
        arr = np.stack([gdm*0, gdm*readout], axis=-1)
    displacement_image = sitk.GetImageFromArray(arr, isVector=True)

    jac = sitk.DisplacementFieldJacobianDeterminant(displacement_image)
    tx = sitk.DisplacementFieldTransform(displacement_image)
    ref = sitk.GetImageFromArray(ima*0)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref)
    # sitkNearestNeighbor, sitk.sitkLinear
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(tx)

    new_ima = resampler.Execute(sitk.GetImageFromArray(ima))
    new_ima = sitk.GetArrayFromImage(new_ima)
    jac_np = sitk.GetArrayFromImage(jac)
    return new_ima*jac_np


def apply_gdm_3d(ima, gdm, readout=1, AP_RL='AP'):

    if AP_RL == 'AP':
        arr = np.stack([gdm*0, gdm*readout, gdm*0], axis=-1)
    else:
        arr = np.stack([gdm*0, gdm*0, gdm*readout], axis=-1)
    displacement_image = sitk.GetImageFromArray(arr, isVector=True)

    jac = sitk.DisplacementFieldJacobianDeterminant(displacement_image)
    tx = sitk.DisplacementFieldTransform(displacement_image)
    ref = sitk.GetImageFromArray(ima*0)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref)
    # sitkNearestNeighbor, sitk.sitkLinear
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(tx)

    new_ima = resampler.Execute(sitk.GetImageFromArray(ima))
    new_ima = sitk.GetArrayFromImage(new_ima)
    jac_np = sitk.GetArrayFromImage(jac)
    return new_ima*jac_np



def gernerate_gdm(gdm_mode, session, orig_data, b0_index, resample=True):

    zoom = orig_data.header.get_zooms()[0:3]
    if len(orig_data.shape)>3:
        vol = orig_data.get_fdata()[...,b0_index]
    else:
        vol = orig_data.get_fdata()
    vol[vol<0] = 0
    
    if resample:
        resample_nii = resample_to_new_resolution(nib.Nifti1Image(vol, orig_data.affine), target_resolution=(1.7, 1.7, 1.7), target_shape=None, interpolation='continuous')
        vol_resize = resample_nii.get_fdata()
        vol_resize = vol_resize / np.max(vol_resize)
    else:
        vol_resize = vol / np.max(vol)
    
    image = vol_resize[None, ...][None, ...]

    #sigmoid = session.run(None, {"modelInput": image.astype(np.float64)})[0]

    logits = predict(session, image)

    if resample:
        df_map = resample_to_new_resolution(nib.Nifti1Image(logits[0, 0, ...], resample_nii.affine), target_resolution=zoom, target_shape=vol.shape, interpolation='linear').get_fdata() * 1.7 / zoom[1]
    else:
        df_map = logits[0, 0, ...]

    df_map_f = np.array(df_map*0, dtype='float64')
    for nslice in np.arange(df_map.shape[2]):
        df_map_slice = gaussian_filter(df_map[..., nslice], sigma=1.5).astype('float64')
        df_map_f[..., nslice] = df_map_slice
    gdm_pred = df_map_f

    

    return gdm_pred
