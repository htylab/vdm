import sys
import os
from os.path import join
import argparse
from distutils.util import strtobool
import glob
from scipy.io import savemat
import nibabel as nib
import numpy as np
import time

import lib_gdm as gdm
import lib_tool as tigertool



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('input',  type=str, nargs='+', help='Path to the input image, can be a folder for the specific format(nii.gz)')
    parser.add_argument('-o', '--output', default=None, help='File path for output image, default: the directory of input files')
    parser.add_argument('-b0', '--b0_index', default=None, type=str, help='The index of b0 slice or the .bval file, default: 0 (the first slice)')
    parser.add_argument('-n', '--no_resample', action='store_true', help='Don\'t resample to 1.7x1.7x1.7mm3')
    parser.add_argument('-m', '--dmap', action='store_true', help='Producing the virtual displacement map')
    parser.add_argument('-g', '--gpu', action='store_true', help='Using GPU')
    parser.add_argument('-model', '--model', default='gdm_gan_v001_fold0', type=str, help='Select specific model')

    args = parser.parse_args() 

    input_file_list = args.input
    if os.path.isdir(args.input[0]):
        input_file_list = glob.glob(join(args.input[0], '*.nii'))
        input_file_list += glob.glob(join(args.input[0], '*.nii.gz'))

    elif '*' in args.input[0]:
        input_file_list = glob.glob(args.input[0])

    output_dir = args.output
    
    if args.b0_index is None:
        b0_index = 0
    elif os.path.exists(args.b0_index.replace('.bval', '') + '.bval'):
        b0_index = gdm.get_b0_slice(args.b0_index.replace('.bval', '') + '.bval')
    else:
        b0_index = int(args.b0_index)
        
    resample = (not args.no_resample)

    print('Total nii files:', len(input_file_list))

    
    model_name = tigertool.get_model(args.model)



    for f in input_file_list:

        print('Predicting:', f)
        t = time.time()
        input_data = gdm.read_file(model_name, f)
        gdmi, gdmap = gdm.run(model_name, input_data, b0_index, GPU=args.gpu, resample=resample)
        
        if output_dir is None:
            f_output_dir = os.path.dirname(os.path.abspath(f))
        else:
            f_output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)
        
        gdm.write_file(model_name, f, f_output_dir, gdmi)
        
        if args.dmap:
            gdm.write_file(model_name, f, f_output_dir, gdmap, postfix='gdm')

        print('Processing time: %d seconds' % (time.time() - t))    

if __name__ == "__main__":
    main()
