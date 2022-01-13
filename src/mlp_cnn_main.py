# by Yang Haibo(2022/1/11)
# yanghaibo4869@gmail.com

import os
import time
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import nibabel as nib
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, normalized_root_mse

from preprocessing import patch_test_img, merge_test_img
from data import *
import mlp_cnn

'''
This code is used to train and valid the network.
It can also save the valid results as .csv format and save the model weights as .pkl format. 

Part of the code can be found in
https://github.com/Deep-Imaging-Group/RED-WGAN/blob/master/model5.py
'''

# train the network in a batch
def _train_generator(free_img, noised_img, G_optimizer, generatornet,train=True):
    z = Variable(noised_img)
    real_img = Variable(free_img, requires_grad=False)

    if torch.cuda.is_available():
        z = z.cuda()
        real_img = real_img.cuda()

    G_optimizer.zero_grad()

    criterion_mse = nn.MSELoss()

    fake_img = generatornet(z)
    mse_loss = criterion_mse(fake_img, real_img)
    if train:
        (mse_loss).backward(retain_graph=True)

    if train:
        G_optimizer.step()
    return mse_loss.data.item()

#  valid the network in valid dataset after one epoch
def test(epoch, batch_size,valid_num_s,valid_num_e,validDataloader,G_optimizer,generatornet,v,level,lr):
    timestr = time.strftime("%H:%M:%S", time.localtime())
    print(timestr)

    total_mse_loss = 0
    batch_num = 0
    for _, batch in enumerate(validDataloader):
        free_img = batch["free_img"]
        noised_img = batch["noised_img"]

        loss = _train_generator(free_img, noised_img, G_optimizer,generatornet,train=False)
        total_mse_loss += loss
        batch_num += 1
    mse_loss = total_mse_loss / batch_num
    print("%s Epoch: %d lr: %.10f Test Loss mse-loss: %.4f" %
            (v, epoch, G_optimizer.defaults["lr"], mse_loss))
    _psnr2, _ssim2 = compute_quality(batch_size,valid_num_s,valid_num_e,level,v,lr,generatornet)
    save_loss((mse_loss),v)

    return _psnr2, _ssim2

# calculate the average PSNR and SSIM of valid dataset.
def compute_quality(batch_size,valid_num_s,valid_num_e,level,v,lr,generatornet):
    psnr1 = 0
    psnr2 = 0
    mse1 = 0
    mse2 = 0
    ssim1 = 0
    ssim2 = 0
    _psnr1 = 0
    _psnr2 = 0
    _mse1 = 0
    _mse2 = 0
    _ssim1 = 0
    _ssim2 = 0
    for i in range(valid_num_s, valid_num_e+1):
        free_nii = nib.load("../data/dataset/Free/%d.nii.gz" % i)
        noised_nii = nib.load(
            "../data/dataset/noise_%d/%d.nii.gz" % (level, i))

        free_img = free_nii.get_fdata()[40:-40, :, :].astype(np.int16)
        noised_img = noised_nii.get_fdata()[40:-40, :, :].astype(np.int16)
        noised_img_non_cut = noised_nii.get_fdata()[:, :, :].astype(np.int16)
        patchs, row, col = patch_test_img(noised_img_non_cut)
        denoised_img = merge_test_img(
            denoising(patchs, batch_size,generatornet), row, col).astype(np.int16)

        psnr1 += peak_signal_noise_ratio(free_img, noised_img,data_range=4096)
        psnr2 += peak_signal_noise_ratio(free_img, denoised_img,data_range=4096)

        mse1 += normalized_root_mse(free_img, noised_img)
        mse2 += normalized_root_mse(free_img, denoised_img)

        ssim1 += structural_similarity(free_img, noised_img,
                                data_range=4096, multichannel=True)
        ssim2 += structural_similarity(free_img, denoised_img,
                                data_range=4096, multichannel=True)

        max = np.max(free_img)
        _psnr1 += peak_signal_noise_ratio(free_img, noised_img,data_range=max)
        _psnr2 += peak_signal_noise_ratio(free_img, denoised_img,data_range=max)

        _mse1 += normalized_root_mse(free_img, noised_img)
        _mse2 += normalized_root_mse(free_img, denoised_img)

        _ssim1 += structural_similarity(free_img, noised_img,
                                data_range=max, multichannel=True)
        _ssim2 += structural_similarity(free_img, denoised_img,
                                data_range=max, multichannel=True)
    sum_num=round(1/(valid_num_e-valid_num_s+1),2)
    psnr1 *= sum_num
    psnr2 *= sum_num
    mse1 *= sum_num
    mse2 *= sum_num
    ssim1 *= sum_num
    ssim2 *= sum_num

    _psnr1 *= sum_num
    _psnr2 *= sum_num
    _mse1 *= sum_num
    _mse2 *= sum_num
    _ssim1 *= sum_num
    _ssim2 *= sum_num
    with open("./loss/" + v + "psnr.csv", "a+") as f:
        f.write("%f,%f,%f,%f,%f,%f\n" %
                (_psnr1, _psnr2, _ssim1, _ssim2, _mse1, _mse2))
    timestr = time.strftime("%H:%M:%S", time.localtime())
    with open("./loss/" + v + "psnr_4096.csv", "a+") as f:
        f.write("%s: %.10f,%f,%f,%f,%f,%f,%f\n" %
                (timestr, lr, psnr1, psnr2, ssim1, ssim2, mse1, mse2))
    print("psnr: %f,%f,ssim: %f,%f,mse:%f,%f\n" %
            (_psnr1, _psnr2, _ssim1, _ssim2, _mse1, _mse2))
    
    return _psnr2, _ssim2

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

# save loss as .csv format in folder named 'loss'
def save_loss(loss,v):
    value = ""
    value = value + str(loss) + ","
    value += "\n"
    with open("./loss/" + v + ".csv", "a+") as f:
        f.write(value)

# save trained weights as .pkl format in folder named 'model'
def save_model(save_dir,generatornet,v):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(generatornet.state_dict(),
                save_dir + "G_" + v + ".pkl")


def main():
    parser = argparse.ArgumentParser(description="train and valid para")

    parser.add_argument('-cu','--cuda_num', default=' 0 ')

    parser.add_argument('-l','--level', default=' 15 ',type=int)
    parser.add_argument('-b','--batch_size', default=' 100 ',type=int)
    parser.add_argument('-e','--epochs', default=' 1000 ',type=int)
    parser.add_argument('-lr','--learn_rates', default=' 5e-4 ',type=float)

    parser.add_argument('-te','--train_num_e', default=' 100 ',type=int)
    parser.add_argument('-vs','--valid_num_s', default=' 101 ',type=int)
    parser.add_argument('-ve','--valid_num_e', default=' 110 ',type=int)

    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_num
    batch_size=args.batch_size
    level=args.level
    epochs=args.epochs
    lr =args.learn_rates

    train_num_e=args.train_num_e
    valid_num_s=args.valid_num_s
    valid_num_e=args.valid_num_e


    v = "mlp_cnn_%d" % level

    save_dir = "./model/" + v + "/"

    dataloader, validDataloader=load_data(batch_size, level,train_num_e,valid_num_s,valid_num_e)

    generatornet = mlp_cnn.generator.GeneratorNet(8, 1536)

    G_optimizer = optim.Adam(
                generatornet.parameters(), lr=lr, betas=(0.5, 0.9))

    if torch.cuda.is_available():
        generatornet = nn.DataParallel(generatornet)
        generatornet.cuda()

    # estimate the end of convergence of the algorithm
    invari = 0
    for epoch in range(0, epochs):
        _psnr2, _ssim2 = test(epoch, batch_size,valid_num_s,valid_num_e,validDataloader,G_optimizer,generatornet,v,level,lr)
        if epoch == 0:
            best_p = _psnr2
            best_s = _ssim2
        if epoch > 0:
            if _psnr2 > best_p and _ssim2 > best_s:
                best_p = _psnr2
                best_s = _ssim2
                save_model(save_dir,generatornet,v)
                with open("./loss/" + v + "best_ps.csv", "a+") as f:
                        f.write("%f,%f,%f\n" %
                                (epoch, best_p, best_s))
                invari = 0
            invari = invari + 1 

        if invari == 5 and lr > 1e-16:
            G_optimizer.defaults["lr"] *= 0.1
            lr *= 0.1
        if invari == 15 and lr > 1e-16:
            G_optimizer.defaults["lr"] *= 0.1
            lr *= 0.1
        if invari == 50:
            break
    
        for _, batch in enumerate(dataloader):
            free_img = batch["free_img"]
            noised_img = batch["noised_img"]
            _train_generator(free_img, noised_img,G_optimizer,generatornet)

if __name__ == '__main__':
    main()