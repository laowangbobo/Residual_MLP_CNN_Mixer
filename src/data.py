# by Yang Haibo(2022/1/11)
# yanghaibo4869@gmail.com

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

'''
This code is used to read train dataset and valid dataset and put them into dataloader.

The original code can be found in
https://github.com/Deep-Imaging-Group/RED-WGAN/blob/master/data.py
'''

# read train dataset
class MRIDataset(Dataset):
    def __init__(self, level,num_e):
        print("Start reading trainning dataset...")
        print("../data/patchs16_16_%d" % level)
        free_mri_set = np.load("../data/patchs16_16_%d/free/1.npy" % level)
        noised_mri_set = np.load("../data/patchs16_16_%d/noised/1.npy" % level)
        for i in range(2, num_e+1):
            path1 = "../data/patchs16_16_%d/free/%d.npy" % (level, i)
            path2 = "../data/patchs16_16_%d/noised/%d.npy" % (level, i)
            free_temp = np.load(path1)
            noised_temp = np.load(path2)

            free_mri_set = np.append(free_mri_set, free_temp, axis=0)
            noised_mri_set = np.append(noised_mri_set, noised_temp, axis=0)
        self.free_mri_set = free_mri_set
        self.noised_mri_set = noised_mri_set
        self.total = self.free_mri_set.shape[0]
        self.current_patch = 1
        print(self.free_mri_set.shape)
        print("End reading trainning dataset...")

    def __len__(self):
        # return 90000
        return self.total

    def __getitem__(self, index):
        free_img = torch.from_numpy(self.free_mri_set[index]).float()
        noised_img = torch.from_numpy(self.noised_mri_set[index]).float()
        return {"free_img": free_img, "noised_img": noised_img}

# read valid dataset
class MRIValidDataset(Dataset):
    def __init__(self, level,num_s,num_e):
        print("Start reading testing dataset...")
        print("../data/patchs16_16_%d" % level)
        free_mri_set = np.load("../data/patchs16_16_%d/free/%d.npy" %(level, num_s))
        noised_mri_set = np.load("../data/patchs16_16_%d/noised/%d.npy" %(level, num_s))
        for i in range(num_s+1, num_e+1):
            path1 = "../data/patchs16_16_%d/free/%d.npy" % (level, i)
            path2 = "../data/patchs16_16_%d/noised/%d.npy" % (level, i)
            free_temp = np.load(path1)
            noised_temp = np.load(path2)

            free_mri_set = np.append(free_mri_set, free_temp, axis=0)
            noised_mri_set = np.append(noised_mri_set, noised_temp, axis=0)
        self.free_mri_set = free_mri_set
        self.noised_mri_set = noised_mri_set
        self.total = self.free_mri_set.shape[0]
        self.current_patch = 1
        print(self.free_mri_set.shape)
        print("End reading testing dataset...")

    def __len__(self):
        # return 90000
        return self.total

    def __getitem__(self, index):
        free_img = torch.from_numpy(self.free_mri_set[index]).float()
        noised_img = torch.from_numpy(self.noised_mri_set[index]).float()
        return {"free_img": free_img, "noised_img": noised_img}

# put dataset into dataloader of pytorch
def load_data(batch_size, level,train_num_e,valid_num_s,valid_num_e):
    dataset = MRIDataset(level,train_num_e)
    # self.dataset = MRIDataset10125()
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True)

    validDataset = MRIValidDataset(level,valid_num_s,valid_num_e)
    validDataloader = DataLoader(
        validDataset, batch_size=batch_size, shuffle=True)
    
    return dataloader, validDataloader


