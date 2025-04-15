# Virtual Displacement Mapping
#### This repo is for the methods described in the paper.
Kuo CC, Huang TY, Lin YR, Chuang TC, Tsai SY, Chung HW, “Referenceless correction of EPI distortion with virtual displacement mapping” (2023), 


* This repo is the version of the paper and only for Pytorch or a stand-alone exe.
* For updated version, please visit: https://github.com/htylab/tigerepi

# The GDM pipeline for DTI distortion correction 

## Background
This package provides GDM distortion correction method for diffusion tensor images


## Tutorial using GDM for EPI Distortion Correction

### Download package

    https://github.com/htylab/gdm/releases/main.zip 

## Usage

### As a command line tool:

- Clone this repository:
```bash
git clone https://github.com/htylab/gdm
cd ./gdm/gdm
```
- Run the predict code:
```
python predict.py INPUT -o OUTPUT
```
If INPUT points to a file, the file will be processed. If INPUT points to a directory, the directory will be searched for the specific format(nii.gz).
OUTPUT is the output directory.

- For additional options type:
```
python predict.py -h
```


### Exe module:
- Windows:
```bash
.\gdm.exe INPUT -o OUTPUT
```

For additional options type:
```bash
.\gdm.exe -h
```

- Linux or Mac:
```bash
./gdm INPUT -o OUTPUT
```

For additional options type:
```bash
./gdm -h
```
