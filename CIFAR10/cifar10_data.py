# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for LACE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import glob
import pickle
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class Cifar10LatentDataset(Dataset):

    def __init__(self, root_path, x_space, train=True, **kwargs):
        self.x_space = x_space

        # data
        if x_space == 'cifar10_i':
            data_fns = sorted(glob.glob(os.path.join(root_path, 'images', '*')))
            if len(data_fns) == 0:
                raise IOError('No image files found in the specified path')
            if train:
                data_fns_selected = data_fns[:-10000]  # First 50000 samples
            else:
                data_fns_selected = data_fns[-10000:]  # Last 10000 samples

            self.num_data = len(data_fns_selected)
            print(f'Dataset size (cifar10_i, train: {train}): {self.num_data}')
            self.data = data_fns_selected

        elif x_space == 'cifar10_z':
            z_latent = pickle.load(open(f'{root_path}/z_latent.pickle', 'rb'))
            z = z_latent['z_latent'][0]  # numpy, 60000x512
            if train:
                z_selected = z[:-10000]  # First 50000 samples
            else:
                z_selected = z[-10000:]  # Last 10000 samples

            self.num_data = z_selected.shape[0]
            print(f'Dataset size (cifar10_z, train: {train}): {self.num_data}')
            self.data = z_selected

        elif x_space == 'cifar10_w':
            w_latent = pickle.load(open(f'{root_path}/ws_latent.pickle', 'rb'))
            w = w_latent['ws_latent'][0][:, 0, :]  # numpy, 60000x512
            if train:
                w_selected = w[:-10000]  # First 50000 samples
            else:
                w_selected = w[-10000:]  # Last 10000 samples

            self.num_data = w_selected.shape[0]
            print(f'Dataset size (cifar10_w, train: {train}): {self.num_data}')
            self.data = w_selected

        else:
            raise NotImplementedError

        # label
        pred = pickle.load(open(f'{root_path}/pred.pickle', 'rb'))
        pred_np = pred['pred'][0]  # numpy, (60000,)
        if train:
            pred_selected = pred_np[:-10000]  # First 50000 samples
        else:
            pred_selected = pred_np[-10000:]  # Last 10000 samples
        assert len(pred_selected) == self.num_data
        self.labels = pred_selected

        # transforms for image input only
        if train:
            self.transform = transforms.Compose(
                [transforms.Pad(4, padding_mode="reflect"),
                 transforms.RandomCrop(32),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize((.5, .5, .5), (.5, .5, .5))])
        else:
            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((.5, .5, .5), (.5, .5, .5))])

    def __len__(self):
        return self.num_data

    def __getitem__(self, i):
        if self.x_space == 'cifar10_i':
            sample = self.transform(Image.open(self.data[i]))
        elif self.x_space == 'cifar10_z':
            sample = torch.tensor(self.data[i])
        elif self.x_space == 'cifar10_w':
            sample = torch.tensor(self.data[i])
        else:
            raise NotImplementedError

        label = torch.tensor(self.labels[i])
        return sample.float(), label


class Cifar10GenDataset(Dataset):

    def __init__(self, root_path, **kwargs):

        self.images_fn = sorted(glob.glob(os.path.join(root_path, 'images', '*')))

        if len(self.images_fn) == 0:
            raise IOError('No image files found in the specified path')
        self.num_data = len(self.images_fn)
        print('Dataset size (cifar-10-gen): ', self.num_data)

        self.transform = transforms.Compose([
            transforms.ToTensor(),  # [0, 1] and [C, H, W]
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    def __len__(self):
        return self.num_data

    def __getitem__(self, i):
        img = np.array(Image.open(self.images_fn[i]))
        if img.ndim != 3:
            raise IOError('Image should be in the RGB format')
        return torch.tensor(i), self.transform(img)
