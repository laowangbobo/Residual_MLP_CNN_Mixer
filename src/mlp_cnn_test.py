# by Yang Haibo(2022/1/11)
# yanghaibo4869@gmail.com


import os
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import nibabel as nib
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, normalized_root_mse

from preprocessing import patch_test_img, merge_test_img
import mlp_cnn


'''
This code is used to test the network.
It can calculate the average PSNR and SSIM of test dataset and save the denoised test images.

Part of the code can be found in
https://github.com/Deep-Imaging-Group/RED-WGAN/blob/master/model5.py
'''

# put noising image into network after training.
def denoising(patchs, batch_size,generatornet):
    n, h, w, d = patchs.shape
    denoised_patchs = []
    for i in range(0, n, batch_size):
        batch = patchs[i:i + batch_size]
        batch_size = batch.shape[0]
        x = np.reshape(batch, (batch_size, 1, w, h, d))
        x = x.transpose(0, 1, 4, 2, 3)
        if torch.cuda.is_available():
            x = Variable(torch.from_numpy(x).float()).cuda()
        else:
            x = Variable(torch.from_numpy(x).float())
        y = generatornet(x)
        denoised_patchs.append(y.cpu().data.numpy())
    denoised_patchs = np.vstack(denoised_patchs)
    denoised_patchs = np.reshape(denoised_patchs, (n, d, h, w))

    denoised_patchs = denoised_patchs.transpose(0, 2, 3, 1)
    return denoised_patchs


# calculate the average PSNR and SSIM of test dataset.
def compute_quality(batch_size,test_num_s, test_num_e,level,generatornet):
    psnr1 = 0
    psnr2 = 0
    mse1 = 0
    mse2 = 0
    ssim1 = 0
    ssim2 = 0
    
    for i in range(test_num_s, test_num_e+1):
        free_nii = nib.load("../data/dataset/Free/%d.nii.gz" % i)
        noised_nii = nib.load(
            "../data/dataset/noise_%d/%d.nii.gz" % (level, i))

        free_img = free_nii.get_fdata()[40:-40, :, :].astype(np.int16)
        noised_img = noised_nii.get_fdata()[40:-40, :, :].astype(np.int16)
        noised_img_non_cut = noised_nii.get_fdata()[:, :, :].astype(np.int16)
        patchs, row, col = patch_test_img(noised_img_non_cut)
        denoised_img = merge_test_img(
            denoising(patchs, batch_size,generatornet), row, col).astype(np.int16)
        

        max = np.max(free_img)
        _psnr1 = peak_signal_noise_ratio(free_img, noised_img,data_range=max)
        _psnr2 = peak_signal_noise_ratio(free_img, denoised_img,data_range=max)

        _mse1 = normalized_root_mse(free_img, noised_img)
        _mse2 = normalized_root_mse(free_img, denoised_img)

        _ssim1 = structural_similarity(free_img, noised_img,
                                data_range=max, multichannel=True)
        _ssim2 = structural_similarity(free_img, denoised_img,
                                data_range=max, multichannel=True)

        
        psnr1 += _psnr1
        psnr2 += _psnr2

        mse1 += _mse1
        mse2 += _mse2

        ssim1 += _ssim1
        ssim2 += _ssim2

        print(_psnr1,_psnr2,_mse1,_mse2,_ssim1,_ssim2)
    psnr1 *= 0.1
    psnr2 *= 0.1
    mse1 *= 0.1
    mse2 *= 0.1
    ssim1 *= 0.1
    ssim2 *= 0.1

    print(psnr1,psnr2,mse1,mse2,ssim1,ssim2)

# load trained model weights into network
def load_model(load_dir, level,generatornet):

    generatornet.load_state_dict(
        torch.load(
            load_dir+'mlp_cnn_%d/G_mlp_cnn_%d.pkl'%(level,level),
            map_location='cpu'))

def main():
    parser = argparse.ArgumentParser(description="train and valid para")

    parser.add_argument('-cu','--cuda_num', default=' 0 ')

    parser.add_argument('-l','--level', default=' 15 ',type=int)
    parser.add_argument('-b','--batch_size', default=' 100 ',type=int)
  
    parser.add_argument('-vs','--test_num_s', default=' 111 ',type=int)
    parser.add_argument('-ve','--test_num_e', default=' 120 ',type=int)

    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_num
    batch_size=args.batch_size
    level=args.level

    test_num_s=args.test_num_s
    test_num_e=args.test_num_e

    generatornet = mlp_cnn.generator.GeneratorNet(8, 1536)

    if torch.cuda.is_available():
        generatornet = nn.DataParallel(generatornet)     
        generatornet.cuda()
    
    load_dir="./model/"
    load_model(load_dir, level,generatornet)
    compute_quality(batch_size,test_num_s, test_num_e,level,generatornet)

    # save denoised test image into folder named 'result'
    for i in range(test_num_s, test_num_e+1):

        nii_img = nib.load("../data/dataset/noise_%d/%d.nii.gz" % (level, i))
        x = nii_img.get_fdata()
        patchs, row, col = patch_test_img(x)
        print(patchs.shape)
        denoised_img = merge_test_img(denoising(patchs,batch_size,generatornet), row, col)
        print(denoised_img.shape)
        denoised_img = denoised_img.astype(np.int16)

        denoised_image = nib.Nifti1Image(
            denoised_img, nii_img.affine, nii_img.header)
        nib.save(denoised_image, "./result/%d_mlp_cnn_%d.nii.gz" % (level, i))

if __name__ == '__main__':
    main()