# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for LACE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import time
import random
from functools import partial

import torch
from torch.utils.data import DataLoader
import torchvision as tv

import numpy as np
from itertools import product
import pickle

import utils
from metfaces_data import MetFacesLatentDataset, get_att_name_list, get_n_classes_list


# ----------------------------------------------------------------------------


class Sampling():
    def __init__(self, mode, att_names, batch_size, latent_dim, ccf, device, root_path, save_path, use_z_init=False):
        self.mode = mode
        self.att_names = att_names
        self.batch_size = batch_size
        self.latent_dim = latent_dim

        self.ccf = ccf
        self.device = device
        self.root_path = root_path
        self.save_path = save_path

        sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
        self.plot = lambda p, x: tv.utils.save_image(torch.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))

        self.use_z_init = use_z_init

        n_classes_list = get_n_classes_list(self.att_names)
        self.seq_len = len(n_classes_list)

    def _sample_batch(self, sampler, ys, z_init=None, save_path=None, z_anchor=None, seq_indices=[],
                      is_return_z=False, reweight=True, dis_temp=1.):
        z_init_cloned = z_init.clone() if z_init is not None else None

        start_sample_time = time.time()
        z_sampled = sampler(ys=ys, z_init=z_init_cloned, save_path=save_path, z_anchor=z_anchor,
                            seq_indices=seq_indices, reweight=reweight, dis_temp=dis_temp)
        g_z_sampled = self.ccf.g(z_sampled)
        img = self.ccf.generate_images(g_z_sampled, seq_indices=seq_indices, update_ws_prev=True)
        sample_time = time.time() - start_sample_time

        if not is_return_z:
            return img, sample_time
        else:
            return img, sample_time, z_sampled



class ConditionalSampling(Sampling):
    def __init__(self, att_names, sampler, batch_size, latent_dim, ccf, device, root_path, save_path, ode_kwargs,
                 ld_kwargs, sde_kwargs, every_n_plot=5, every_n_print=5, use_z_init=False, mode='cond'):
        super().__init__(mode, att_names, batch_size, latent_dim, ccf, device, root_path, save_path, use_z_init)

        self.sampler = partial(sampler, ccf=ccf, device=device, plot=self.plot,
                               every_n_plot=every_n_plot, every_n_print=every_n_print,
                               **ode_kwargs, **ld_kwargs, **sde_kwargs)

    def get_samples(self, att_vals=None, lo=0.25, hi=0.75, discrete_num=3, trim_num=3, z_init=None, subdir=''):

        if att_vals is None:  # discrete uniformly the continuous value in [lo, hi]
            save_path = os.path.join(self.save_path, 'cond')

            n_classes_list = get_n_classes_list(self.att_names)
            ys_list = nclasses_to_ys(n_classes_list, lo, hi, discrete_num, trim_num)

        else:  # some user-specified values given the attribute names
            save_path = os.path.join(self.save_path, 'cond_use_case')
            if subdir != '':
                save_path = os.path.join(save_path, subdir)

            assert isinstance(att_vals, str), att_vals
            att_val_list = list(map(float, att_vals.split('_')))
            assert len(att_val_list) == self.seq_len
            ys_list = [att_val_list] * trim_num

        os.makedirs(save_path, exist_ok=True)

        if self.use_z_init and z_init is None:  # z_init will be set the same for with different ys
            z_init = torch.FloatTensor(self.batch_size, self.latent_dim).normal_(0, 1).to(self.device)

        cnt = 0
        for ys in ys_list:
            ys = torch.tensor([ys]).repeat(self.batch_size, 1).to(self.device)

            img, sample_time = self._sample_batch(self.sampler, ys, z_init, save_path=save_path)
            attr_vals = '_'.join([f"{attr_val:.3f}" for attr_val in ys[0].cpu().numpy()])
            print(f'class {attr_vals}, sampling time: {sample_time}')

            self.plot(f'{save_path}/samples_{self.att_names}_class{attr_vals}_cnt{cnt}.png', img)
            cnt += 1


# ----------------------------------------------------------------------------


class SequentialEditing(Sampling):
    def __init__(self, att_names, sampler, batch_size, latent_dim, ccf, device, root_path, save_path,
                 ode_kwargs, ld_kwargs, sde_kwargs, every_n_plot=5, every_n_print=5, use_z_init=False,
                 update_z_init=True, reg_z=0, reg_id=0, reg_logits=0, seq_len=0, seq_method='Method1',
                 reweight=True, dis_temp=1., mode='seq_edit'):
        super().__init__(mode, att_names, batch_size, latent_dim, ccf, device, root_path, save_path, use_z_init)

        self.update_z_init = update_z_init
        self.seq_len = seq_len
        self.seq_method = seq_method
        self.reweight = reweight
        self.dis_temp = dis_temp

        self.sampler = partial(sampler, ccf=ccf, device=device, plot=self.plot,
                               every_n_plot=every_n_plot, every_n_print=every_n_print,
                               reg_z=reg_z, reg_id=reg_id, reg_logits=reg_logits,
                               **ode_kwargs, **ld_kwargs, **sde_kwargs)

        print(f'mode: {mode}, seq_method: {seq_method}, reweight: {reweight}, dis_temp: {dis_temp}')

    def get_samples(self, att_vals=None, att_vals_init=None, cond_first=True, lo=0.25, hi=0.75,
                    discrete_num=3, trim_num=3, zinit_seed=1):

        if att_vals is None or att_vals_init is None:  # discrete uniformly the continuous value in [lo, hi]
            save_path = os.path.join(self.save_path, 'seq_edit')

            n_classes_list = get_n_classes_list(self.att_names)[:self.seq_len]
            ys_list = nclasses_to_ys(n_classes_list, lo, hi, discrete_num, trim_num)

        else:  # some user-specified values given the attribute names
            save_path = os.path.join(self.save_path, 'seq_edit_use_case')

            assert isinstance(att_vals, str) and isinstance(att_vals_init, str)
            att_val_list = list(map(float, att_vals.split('_')))
            att_val_list_init = list(map(float, att_vals_init.split('_')))
            assert len(att_val_list) == len(att_val_list_init) == self.seq_len

            ys_list, ys_list_cond = [att_val_list], [att_val_list_init]

        os.makedirs(save_path, exist_ok=True)

        cnt = 0
        for idx, ys in enumerate(ys_list):
            ys = torch.tensor([ys]).repeat(self.batch_size, 1).to(self.device)

            save_path_seq = os.path.join(save_path, '_'.join(map(str, ys[0].cpu().numpy())))
            os.makedirs(save_path_seq, exist_ok=True)

            z_init = None
            if self.use_z_init:  # z_init will be set the same for seq editing
                z_init = torch.FloatTensor(self.batch_size, self.latent_dim).\
                    normal_(0, 1, generator=torch.manual_seed(zinit_seed)).to(self.device)

            # ---------------------------- [start] conditional sampling first ----------------------------
            if (att_vals is not None and att_vals_init is not None) and cond_first:
                print('not uniformly sampled and cond first (w/o discrete temperature)...')
                ys_cond = torch.tensor([ys_list_cond[idx]]).repeat(self.batch_size, 1).to(self.device)

                z_init_cloned = z_init.clone() if z_init is not None else None
                z_init = self.sampler(ys=ys_cond, seq_indices=list(range(self.seq_len)), z_init=z_init_cloned,
                                      reweight=False)

            g_z_sampled = self.ccf.g(z_init).detach()
            img = self.ccf.generate_images(g_z_sampled, seq_indices=list(range(self.seq_len)), update_ws_prev=True)
            self.plot(f'{save_path_seq}/samples_init.png', img)
            pickle.dump({'Latent': g_z_sampled.cpu().numpy()}, open(f'{save_path_seq}/seq_latent_init.pickle', 'wb'))
            # ---------------------------- [end] conditional sampling first ----------------------------

            z_anchor = z_init
            for i in range(self.seq_len):

                seq_indices = self.get_seq_indices(i)
                print(f'seq_indices: {seq_indices}')
                img, sample_time, z_sampled = self._sample_batch(self.sampler, ys[:, seq_indices], z_init,
                                                                 save_path=save_path_seq, z_anchor=z_anchor,
                                                                 seq_indices=seq_indices, is_return_z=True,
                                                                 reweight=self.reweight, dis_temp=self.dis_temp)
                attr_vals = '_'.join([f"{attr_val:.3f}" for attr_val in ys[0][:i + 1].cpu().numpy()])
                print(f'class {attr_vals}, sampling time: {sample_time}')

                att_names_sub = '_'.join(self.att_names.split('_')[:i + 1])
                self.plot(f'{save_path_seq}/samples_{att_names_sub}_class{attr_vals}_cnt{cnt}.png', img)

                z_anchor = z_sampled
                if self.update_z_init and z_sampled is not None:
                    print('Updating z_init...')
                    z_init = z_sampled
            cnt += 1

    def get_seq_indices(self, i):
        if self.seq_method == 'Method1':
            seq_indices = list(range(i + 1))
        elif self.seq_method == 'Method2':
            seq_indices = [i]
        else:
            raise NotImplementedError
        return seq_indices


# ----------------------------------------------------------------------------


def nclasses_to_ys(n_classes_list, lo, hi, discrete_num, trim_num):
    ys_list = list(product(*[range(n_classes) if n_classes > 1 else list(np.linspace(lo, hi, num=discrete_num))
                             for n_classes in n_classes_list]))
    if 0 < trim_num < len(ys_list):
        ys_list = random.sample(ys_list, trim_num)  # without replacement
    return ys_list


# ----------------------------------------------------------------------------


def test_clf_acc(ccf, att_names, dataset, root_path, batch_size, device):
    att_names = get_att_name_list(att_names, for_model=False)

    if dataset == "metfaces_latent_train":
        dset = MetFacesLatentDataset(root_path, att_names, train=True)
        clf_func = ccf.classify_x
    elif dataset == "metfaces_latent_test":
        dset = MetFacesLatentDataset(root_path, att_names, train=False)
        clf_func = ccf.classify_x
    else:
        raise NotImplementedError

    dload = DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)

    corrects_list, losses_list = [[] for _ in att_names], [[] for _ in att_names]
    n_classes_list = get_n_classes_list(att_names)
    corrects_per_cls_list = [[[] for _ in range(n_classes)] for n_classes in n_classes_list]
    nums_per_cls_list = [[0 for _ in range(n_classes)] for n_classes in n_classes_list]

    for x_p_d, ys in dload:
        x_p_d, ys = x_p_d.to(device), ys.to(device)
        logits_list = clf_func(x_p_d)

        for i, logits in enumerate(logits_list):
            assert n_classes_list[i] == logits.size(1)

            loss = utils.get_loss(logits, ys[:, i], device=device).cpu().numpy()
            losses_list[i].extend(loss)

            correct = utils.get_acc(logits, ys[:, i]).cpu().numpy()
            corrects_list[i].extend(correct)

            if logits.size(1) > 1:  # output acc of each class for discrete attribute
                for c in range(logits.size(1)):
                    acc_c, nums_c = utils.get_acc(logits, ys[:, i], c=c)
                    corrects_per_cls_list[i][c].extend(acc_c.cpu().numpy())
                    nums_per_cls_list[i][c] += nums_c.cpu().numpy()

    for n_classes, att_name, losses, corrects, corrects_per_cls, nums_per_cls in zip(
            n_classes_list, att_names, losses_list, corrects_list, corrects_per_cls_list, nums_per_cls_list):
        loss = np.mean(losses)
        correct = np.mean(corrects)
        if n_classes > 1:  # discrete
            correct_cls_list = [(np.sum(corrects_c) / num_c, num_c)
                                for corrects_c, num_c in zip(corrects_per_cls, nums_per_cls)]
            print(f'attr: {att_name} (disc.), loss: {loss}, correct: {correct}, correct_cls: {correct_cls_list}')
        else:
            print(f'attr: {att_name} (cont.), loss: {loss}, correct: {correct}')


# ----------------------------------------------------------------------------
