# Denoising of 3D MR images using a voxel-wise hybrid residual MLP-CNN model to improve small lesion diagnostic confidence
PyTorch implementation of *Denoising of 3D MR images using a voxel-wise hybrid residual MLP-CNN model to improve small lesion diagnostic confidence*
To appear in 25th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI 2022)
[[publication]]() [[arxiv]]()
If this work is useful to you, please cite our paper:
```
@article
```
## Requirements
- Python 3.9.5
- torch 1.9.1+cu111 
- numpy 1.21.4
- scikit-image 0.18.3
- nibabel 3.2.1
## Main Pipeline
![Main Pipeline](figs/network.png)
## Data Acquisition
Images used in the paper are T2-FLAIR brain MRIs with white matter hyperintensities (WMH) lesions from [UK biobank](http://www.ukbiobank.ac.uk). The dataset is not free, You need to email the official for details.
You can also use other free public dataset of 3D MRI in other modality, such as [IXI Dataset](https://brain-development.org/ixi-dataset/), [ATLAS](https://www.icpsr.umich.edu/web/ADDEP/studies/36684/versions/V3), etc.
## Initialization and Preprocessing
Once you get the data, please convert it into .nii.gz files and put them into ../data/dataset/Free/ fold. Please Name all files in numerical order (like, 1.nii.gz, 2.nii.gz, ...). Please reshape your image matrix into 256 x 256 x 6.
Before you start training, please inital all the folds,  get into ../src/ fold and run the following script.
```
python3 init_files.py
```
After initialization, data preprocessing is needed. You need generate simulated noisy images and reshape each image into patches,  run the following script.
```
python3 preprocessing.py
```
