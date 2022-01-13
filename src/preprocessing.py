# by Yang Haibo(2022/1/11)
# yanghaibo4869@gmail.com


import os
import nibabel as nib
import numpy as np
import argparse

'''
This code is used to generate noisy image and reshape them into patches.
It can also reshape patches into compelete image.

Part of the code can be found in
https://github.com/Deep-Imaging-Group/RED-WGAN/blob/master/preprocessing.py
'''

# add rician noise to free image.
def add_rice_noise(img, snr=10, mu=0.0, sigma=1):
    level = snr * np.max(img) / 100
    size = img.shape
    x = level * np.random.normal(mu, sigma, size=size) + img
    y = level * np.random.normal(mu, sigma, size=size)
    return np.sqrt(x**2 + y**2).astype(np.int16)

# generate and save image with noise.
def generate_noised_mri(noisy_level):
    files = os.listdir("../data/dataset/Free/")
    for file in files:
        nii_img = nib.load("../data/dataset/Free/" + file)
        free_image = nii_img.get_fdata()
        noised_image = add_rice_noise(free_image, snr=noisy_level)
        noised_image = nib.Nifti1Image(noised_image, nii_img.affine, nii_img.header)
        if not os.path.exists("../data/dataset/noise_%d/" % noisy_level):
            os.makedirs("../data/dataset/noise_%d/" % noisy_level)
        nib.save(noised_image, "../data/dataset/noise_%d/%s" % (noisy_level, file))

# reshape image into patches and save them.
def generate_patch(level):
    stride = 10
    size = 16
    depth = 6
    num = 0
    files = os.listdir("../data/dataset/Free/")
    for file in files:
        free_img = nib.load("../data//dataset/Free/" + file).get_fdata()
        noised_img = nib.load("../data/dataset/noise_%d/%s" % (level, file)).get_fdata()
        free_img_set = None
        noised_img_set = None
        height, width, _ = free_img.shape
        for y in range(40, height-size-40+1, stride):
            for x in range(0, width-size+1, stride):
                free_img_temp = free_img[y:y+size, x:x+size].copy().transpose(2, 0, 1)
                noised_img_temp = noised_img[y:y + size,
                                             x:x + size].copy().transpose(2, 0, 1)
                # print(free_img_temp.shape)
                free_img_temp = np.reshape(
                    free_img_temp, (1, 1, depth, size, size))
                noised_img_temp = np.reshape(
                    noised_img_temp, (1, 1, depth, size, size))
                
                if free_img_set is None:
                    free_img_set = free_img_temp
                    noised_img_set = noised_img_temp
                else:
                    free_img_set = np.append(free_img_set, free_img_temp, axis=0)
                    noised_img_set = np.append(noised_img_set, noised_img_temp, axis=0)
        num += 1
        print("-------" + str(num) + "-----------")
        print(noised_img_set.shape)
        print(free_img_set.shape)
        if not os.path.exists("../data/patchs16_16_%d/free/" % level):
            os.makedirs("../data/patchs16_16_%d/free/" % level)
            os.makedirs("../data/patchs16_16_%d/noised/" % level)
        np.save("../data/patchs16_16_%d/free/%d.npy" % (level, num), free_img_set)
        np.save("../data/patchs16_16_%d/noised/%d.npy" % (level, num), noised_img_set)

# reshape image into patches and form them as matrix.
def patch_test_img(img, size=16):
    patchs = []
    height, width, depth = img.shape
    stride = 10
    
    row = 0
    
    for i in range(40, height-size-40+1, stride):
        row += 1
        col = 0
        for j in range(0, width-size+1, stride):
            col += 1
            patchs.append(img[i:i+size, j:j+size,:])
    temp = np.vstack(patchs)
    temp = np.reshape(temp, (-1, size, size, depth))
    return temp, row, col

# reshape patches into a compelete image.
def merge_test_img(patchs, row, col,size=16):
    stride = 10
    for i in range(row):
        for j in range(col):
            if j == 0:
                img = patchs[i*col + j, :, :stride, :]
            if 0 < j < col-1:
                img = np.append(img, patchs[i*col + j, :, :stride, :], axis=1)
            if j == col - 1:
                img = np.append(img, patchs[i*col + j, :, :, :], axis=1)
        if i == 0:
            imgx = img[:stride, :,:]
        if 0< i < row-1:
            imgx = np.append(imgx, img[:stride, :,:], axis=0)
        if i == row-1:
            imgx = np.append(imgx, img[:, :,:], axis=0)
    return imgx


def main():
    parser = argparse.ArgumentParser(description="noise level para")
    parser.add_argument('-l','--level', default=' 15 ',type=int)
    args = parser.parse_args()
    level = args.level
    generate_noised_mri(level)
    generate_patch(level)

if __name__ == '__main__':
    main()

    