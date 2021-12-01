# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for LACE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import time
from functools import partial

import torch
import torch.nn as nn
import torchvision.transforms as tr
from torch.utils.data import DataLoader
import torchvision as tv

import numpy as np
from tqdm import tqdm
import pickle

import utils
from models import DenseNet
from cifar10_data import Cifar10LatentDataset
from metrics.calc_inception import load_patched_inception_v3


# ----------------------------------------------------------------------------


class Sampling():
    def __init__(self, batch_size, latent_dim, n_classes, ccf, device, save_path):
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        self.ccf = ccf
        self.device = device
        self.save_path = save_path

        sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
        self.plot = lambda p, x: tv.utils.save_image(torch.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))

    def _sample_batch(self, sampler, y, save_path=None):

        start_sample_time = time.time()
        z_sampled = sampler(y=y, save_path=save_path)
        g_z_sampled = self.ccf.g(z_sampled)
        img = self.ccf.generate_images(g_z_sampled)
        sample_time = time.time() - start_sample_time

        return img, sample_time

    def _extract_acc_or_feat(self, acc_or_feat, n_samples, clf_model=None, inception=None):
        raise NotImplementedError

    def eval_acc(self, n_samples=1000):
        clf_model = DenseNet(num_classes=self.n_classes, depth=190, growthRate=40, compressionRate=2,
                             dropRate=0).to(self.device)
        clf_model = nn.DataParallel(clf_model)
        load_path = './pretrained/classifiers/cifar10/densenet-bc-L190-k40/model_best.pth.tar'
        print(f"loading model from {load_path}")
        classifier_ckpt_dict = torch.load(load_path)
        clf_model.load_state_dict(classifier_ckpt_dict["state_dict"])
        clf_model.eval()

        start_time = time.time()
        acc_cls = self._extract_acc_or_feat('acc', n_samples=n_samples, clf_model=clf_model)
        print(f'getting {n_samples} accs take time: {time.time() - start_time}')

        acc = np.mean(acc_cls)
        print(f'acc: {acc:.3f}')
        accs = [np.mean(acc_per_cls) for acc_per_cls in np.split(acc_cls, self.n_classes, axis=0)]
        print('acc (class 0-9): ', [f"{amean:.3f}" for amean in accs])

    def eval_fid(self, n_samples=50000, inception_pkl=''):

        inception = nn.DataParallel(load_patched_inception_v3()).to(self.device)
        inception.eval()

        start_time = time.time()
        features = self._extract_acc_or_feat('feat', n_samples=n_samples, inception=inception)
        print(f'getting {n_samples} features take time: {time.time() - start_time}')

        with open(inception_pkl, "rb") as f:
            embeds = pickle.load(f)
            real_mean = embeds["mean"]
            real_cov = embeds["cov"]

        sample_mean = np.mean(features, 0)
        sample_cov = np.cov(features, rowvar=False)

        fid = utils.calc_fid(sample_mean, sample_cov, real_mean, real_cov)

        print(f"fid: {fid:.3f}")


class ConditionalSampling(Sampling):
    def __init__(self, sampler, batch_size, latent_dim, n_classes, ccf, device, save_path, ode_kwargs,
                 ld_kwargs, sde_kwargs, every_n_plot=5):
        super().__init__(batch_size, latent_dim, n_classes, ccf, device, save_path)

        self.sampler = partial(sampler, ccf=ccf, device=device, plot=self.plot, every_n_plot=every_n_plot,
                               **ode_kwargs, **ld_kwargs, **sde_kwargs)

    def get_samples(self):
        save_path = os.path.join(self.save_path, 'cond')
        os.makedirs(save_path, exist_ok=True)

        for i in range(self.n_classes):
            y = torch.tensor([i]).repeat(self.batch_size).to(self.device)

            img, sample_time = self._sample_batch(self.sampler, y, save_path=save_path)
            print(f'class {i}, sampling time: {sample_time}')
            self.plot('{}/samples_class{}.png'.format(save_path, i), img)

    def _extract_acc_or_feat(self, acc_or_feat, n_samples, clf_model=None, inception=None):
        n_samples_cls = n_samples // self.n_classes
        assert self.n_classes * n_samples_cls == n_samples

        batch_size_list = nsamples_to_batches(n_samples_cls, self.batch_size)

        res_batches = []
        for i in range(self.n_classes):

            for bs in tqdm(batch_size_list):
                y = torch.tensor([i]).repeat(bs).to(self.device)
                img, sample_time = self._sample_batch(self.sampler, y)
                print(f'class {i}, sampling time: {sample_time}')

                if acc_or_feat == 'acc':
                    with torch.no_grad():
                        logits = clf_model(img)
                    acc = (logits.max(1)[1] == y).float()
                    res_batches.append(acc)

                else:  # acc_or_feat == 'feat'
                    with torch.no_grad():
                        feat = inception(img)[0].view(img.shape[0], -1)
                    res_batches.append(feat)

        accs_or_feats = torch.cat(res_batches, 0)
        assert accs_or_feats.shape[0] == n_samples

        return accs_or_feats.cpu().numpy()


# ----------------------------------------------------------------------------


def nsamples_to_batches(n_samples, batch_size):
    n_batch = n_samples // batch_size
    resid = n_samples - (n_batch * batch_size)
    res = [resid] if resid > 0 else []
    batch_size_list = [batch_size] * n_batch + res
    return batch_size_list


# ----------------------------------------------------------------------------

def test_clf(ccf, x_space, dataset, root_path, batch_size, device):
    transform = tr.Compose(
        [tr.ToTensor(),
         tr.Normalize((.5, .5, .5), (.5, .5, .5))]
    )

    train = True if dataset == "cifar_train" else False
    if x_space == 'cifar10_i':
        dset = tv.datasets.CIFAR10(root="../data", transform=transform, download=True, train=train)
    else:
        dset = Cifar10LatentDataset(root_path, x_space, train=train)

    dload = DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)

    corrects, losses= [], []
    for x_p_d, y_p_d in tqdm(dload):
        x_p_d, y_p_d = x_p_d.to(device), y_p_d.to(device)
        logits = ccf.classify_x(x_p_d)
        loss = nn.CrossEntropyLoss(reduce=False)(logits, y_p_d).cpu().detach().numpy()
        losses.extend(loss)
        correct = (logits.max(1)[1] == y_p_d).float().cpu().numpy()
        corrects.extend(correct)

    loss = np.mean(losses)
    correct = np.mean(corrects)
    print(f'loss: {loss}, correct: {correct}')


# ----------------------------------------------------------------------------

