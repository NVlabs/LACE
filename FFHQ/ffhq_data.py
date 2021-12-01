# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for LACE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import glob
import pickle
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


att_dict = {'width': [0, 1], 'height': [1, 1], 'smile': [2, 2],
            'pitch': [3, 1], 'roll': [4, 1], 'yaw': [5, 1],
            'gender': [6, 2], 'age': [7, 1], 'beard': [8, 2],
            'glasses': [9, 2], 'bald': [10, 1], 'haircolor': [11, 5],
            'light0': [12, 1],
            'light1': [13, 1],
            'light2': [14, 1],
            'light3': [15, 1],
            'light4': [16, 1],
            'light5': [17, 1],
            'light6': [18, 1],
            'light7': [19, 1],
            'light8': [20, 1]}


def get_ngram_dict(root_path, att_names):
    att_names = get_att_name_list(att_names, for_model=False)
    n_classes_list = get_n_classes_list(att_names)
    att_idxes = np.array([att_dict[att_name][0] for att_name in att_names])

    att_light = pickle.load(open(f'{root_path}/att_light.pickle', 'rb'))
    att_light_np = att_light['att_light'][0][:, att_idxes]  # numpy, (10000,)

    # None in labels to -1
    att_light_np[att_light_np == None] = -1

    # discard samples without labels
    id_eff_samples = np.argwhere(np.all(att_light_np != -1, axis=1)).squeeze()
    att_light_selected = att_light_np[id_eff_samples]

    print(f'att_light_selected.shape: {att_light_selected.shape}')

    bins10 = list(zip(np.arange(0, 1, 0.1), np.arange(0.1, 1.1, 0.1)))
    def bin_att_light(att_light):
        att_light_binned = []
        for n_classes, val in zip(n_classes_list, att_light):
            assert val is not None and val != -1, val
            if n_classes == 1:
                for bin in bins10:
                    if val <= bin[1]:
                        val = (bin[0] + bin[1]) / 2
                        break
            att_light_binned.append(val)
        return tuple(att_light_binned)

    # get ngram dict
    ngram_dict = {}
    for att_light in att_light_selected:
        att_light = bin_att_light(att_light)
        ngram_dict[att_light] = ngram_dict.get(att_light, 0) + 1

    print(f'ngram_dict len: {len(ngram_dict)}, sum: {sum(ngram_dict.values())}')

    return ngram_dict


def get_att_name_list(att_names, for_model=True):
    if not isinstance(att_names, list):
        if for_model:
            att_names = [x for x in att_names.split('_')]
        else:
            att_names = [t for x in att_names.split('_') for t in x.split('-')]

    return att_names


def get_n_classes_list(att_names):
    att_names = get_att_name_list(att_names, for_model=False)
    n_classes_list = [att_dict[att_name][1] for att_name in att_names]
    return n_classes_list


class FFHQLatentDataset(Dataset):

    def __init__(self, root_path, att_names, train=True, **kwargs):
        att_names = get_att_name_list(att_names, for_model=False)

        self.att_names = att_names
        split = 1000
        att_idxes = np.array([att_dict[att_name][0] for att_name in att_names])

        # w
        w_latent = pickle.load(open(f'{root_path}/all_latents.pickle', 'rb'))
        w = w_latent['Latent'][:, 0, 0, :]  # numpy, 10000x512
        if train:
            w_selected = w[:-split]  # First 9000 samples
        else:
            w_selected = w[-split:]  # Last 1000 samples

        # label
        att_light = pickle.load(open(f'{root_path}/att_light.pickle', 'rb'))
        att_light_np = att_light['att_light'][0][:, att_idxes]  # numpy, (10000,)
        if train:
            att_light_selected = att_light_np[:-split]  # First 9000 samples
        else:
            att_light_selected = att_light_np[-split:]  # Last 1000 samples

        # None in labels to -1
        att_light_selected[att_light_selected == None] = -1

        # discard samples without labels (if there exists any -1)
        id_eff_samples = np.argwhere(np.all(att_light_selected != -1, axis=1)).squeeze()

        w_selected = w_selected[id_eff_samples]
        att_light_selected = att_light_selected[id_eff_samples]

        self.num_data = w_selected.shape[0]
        print(f'Dataset size (FFHQ_Latent ({att_names}), train: {train}): {self.num_data}')
        self.data = w_selected
        assert len(att_light_selected) == self.num_data
        self.labels = att_light_selected.astype('float')

    def __len__(self):
        return self.num_data

    def __getitem__(self, i):
        sample = torch.tensor(self.data[i]).float()
        label = torch.tensor(self.labels[i]).float()   # shape is [num_label, 1] if a single attribute
        return sample, label


class FFHQGenDataset(Dataset):

    def __init__(self, root_path, att_names, train=True, res=1024, **kwargs):
        self.res = res

        att_names = get_att_name_list(att_names, for_model=False)

        self.att_names = att_names
        split = 1000
        att_idxes = np.array([att_dict[att_name][0] for att_name in att_names])
        if res == 1024:
            # Data transforms (for fid evaluation)
            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            img_dir = 'images'

        else:  # res == 256
            # Data transforms (for classification)
            self.transform = transforms.Compose(
                [transforms.Resize(224),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.res = 224
            img_dir = 'images_256'

        image_fns = np.array(sorted(glob.glob(f'{root_path}/{img_dir}/*.png')))
        if train:
            image_fns_selected = image_fns[:-split]  # First 9000 samples
        else:
            image_fns_selected = image_fns[-split:]  # Last 1000 samples

        # label
        att_light = pickle.load(open(f'{root_path}/att_light.pickle', 'rb'))
        att_light_np = att_light['att_light'][0][:, att_idxes]  # numpy, (10000,)
        if train:
            att_light_selected = att_light_np[:-split]  # First 9000 samples
        else:
            att_light_selected = att_light_np[-split:]  # Last 1000 samples

        # None in labels to -1
        att_light_selected[att_light_selected == None] = -1

        # discard samples without labels
        id_eff_samples = np.argwhere(np.any(att_light_selected != -1, axis=1)).squeeze()

        image_fns_selected = image_fns_selected[id_eff_samples]
        att_light_selected = att_light_selected[id_eff_samples]

        self.num_data = image_fns_selected.shape[0]
        print(f'Dataset size (FFHQ_Gen ({att_names}), train: {train}): {self.num_data}')
        self.data_path = image_fns_selected
        assert len(att_light_selected) == self.num_data
        self.labels = att_light_selected.astype('float')

    def __len__(self):
        return self.num_data

    def __getitem__(self, i):
        image = Image.open(self.data_path[i])
        image_tensor = self.transform(image)
        label = torch.tensor(self.labels[i]).float()   # shape is [num_label, 1] if a single attribute

        assert image_tensor.shape[-1] == self.res, image_tensor.shape[-1]
        return image_tensor, label


class FFHQDataset(Dataset):

    def __init__(self, root_path, **kwargs):

        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        image_fns = np.array(sorted(glob.glob(f'{root_path}/ffhq/*.png')))

        self.num_data = image_fns.shape[0]
        print(f'Dataset size (FFHQ): {self.num_data}')
        self.data_path = image_fns

    def __len__(self):
        return self.num_data

    def __getitem__(self, i):
        image = Image.open(self.data_path[i])
        image_tensor = self.transform(image)
        label_dummy = torch.zeros([1]).float()   # to keep it consistent with other labeled datasets

        assert image_tensor.shape[-1] == 1024  # res: 1024
        return image_tensor, label_dummy

