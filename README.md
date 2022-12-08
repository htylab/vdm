# tigervdm
#### This repo is for the methods described in the paper.
Kuo CC, Huang TY, “Referenceless correction of EPI distortion with virtual displacement mapping” (2022), 


* This repo is the version of the paper and only for Pytorch or a stand-alone exe.
* For updated version, please visit: https://github.com/htylab/tigervdm

# The VDM pipeline for DTI distortion correction 

## Background
This package provides VDM distortion correction method for diffusion tensor images


## Tutorial using VDM for EPI Distortion Correction

### Download package

    https://github.com/htylab/vdm/releases/main.zip 

## Usage

### As a command line tool:

- Clone this repository:
```bash
git clone https://github.com/htylab/vdm
cd ./vdm/vdm
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
.\vdm.exe INPUT -o OUTPUT
```

For additional options type:
```bash
.\vdm.exe -h
```

- Linux or Mac:
```bash
./vdm INPUT -o OUTPUT
```

For additional options type:
```bash
./vdm -h
```
