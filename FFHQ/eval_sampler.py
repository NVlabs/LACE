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
import torch.nn as nn, torch.nn.functional as functional
from torch.utils.data import DataLoader
import torchvision as tv

import numpy as np
from tqdm import tqdm
from itertools import product
import pickle

import utils
from ffhq_data import FFHQLatentDataset, get_ngram_dict, get_att_name_list, get_n_classes_list
from metrics.id_loss import IDLoss
from metrics.calc_inception import load_patched_inception_v3


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

    def _extract_acc_or_feat(self, acc_or_feat, sample_att_func, n_samples,
                             clf_model=None, clf_indices=[], inception=None):
        raise NotImplementedError

    def eval_acc(self, att_source='uniform', n_samples=1000):
        clf_model, att_names_to_indices = get_clf_model(self.device)
        clf_indices = att_names_to_indices(self.att_names)[:self.seq_len]
        clf_model.eval()
        print('clf_model #params: {}'.format(utils.compute_n_params(clf_model)))
        print('clf_indices: {}'.format(clf_indices))

        sample_att_func = get_attributes_sampler(att_names=self.att_names, att_source=att_source,
                                                 seq_len=self.seq_len, root_path=self.root_path)

        start_time = time.time()
        accs_list = self._extract_acc_or_feat('acc', sample_att_func, n_samples=n_samples,
                                              clf_model=clf_model, clf_indices=clf_indices)
        print(f'getting {n_samples} features and accs take time: {time.time() - start_time}')

        for i, accs in enumerate(accs_list):
            acc_mean = np.mean(accs, 0)
            print(f"[{self.mode}-{i}] acc: ", [f"{amean:.3f}" for amean in acc_mean])

    def eval_fid(self, att_source='training', inception_pkl=''):

        def get_n_samples(inception_pkl):
            inception_pkl_name = os.path.basename(inception_pkl)
            if inception_pkl_name == 'inception_ffhq_gen_1k.pkl':
                return 1000
            else:
                return 50000

        n_samples = get_n_samples(inception_pkl)

        inception = nn.DataParallel(load_patched_inception_v3()).to(self.device)
        inception.eval()
        print('inception #params: {}'.format(utils.compute_n_params(inception.module)))

        sample_att_func = get_attributes_sampler(att_names=self.att_names, att_source=att_source,
                                                 seq_len=self.seq_len, root_path=self.root_path)

        start_time = time.time()
        feats_list = self._extract_acc_or_feat('feat', sample_att_func, n_samples=n_samples,
                                               inception=inception)
        print(f'getting {n_samples} features and accs take time: {time.time() - start_time}')

        with open(inception_pkl, "rb") as f:
            embeds = pickle.load(f)
            real_mean = embeds["mean"]
            real_cov = embeds["cov"]

        for i, features in enumerate(feats_list):
            sample_mean = np.mean(features, 0)
            sample_cov = np.cov(features, rowvar=False)

            fid = utils.calc_fid(sample_mean, sample_cov, real_mean, real_cov)
            print(f"[{self.mode}-{i}] fid: {fid:.3f}")


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

    def _extract_acc_or_feat(self, acc_or_feat, sample_att_func, n_samples,
                             clf_model=None, clf_indices=[], inception=None):
        batch_size_list = nsamples_to_batches(n_samples, self.batch_size)

        res_batches = []
        for bs in tqdm(batch_size_list):
            # sample ys
            ys = sample_att_func()
            ys = torch.tensor([ys]).repeat(bs, 1).to(self.device)

            img, sample_time = self._sample_batch(self.sampler, ys)
            print(f'class {ys[0].cpu().numpy()}, sampling time: {sample_time}')

            if acc_or_feat == 'acc':
                acc_batch = classify_img(img, ys, clf_model, clf_indices)
                res_batches.append(acc_batch)

            else:  # acc_or_feat == 'feat'
                with torch.no_grad():
                    feat = inception(img)[0].view(img.shape[0], -1)
                res_batches.append(feat)

        accs_or_feats = torch.cat(res_batches, 0)
        assert accs_or_feats.size(0) == n_samples

        return [accs_or_feats.cpu().numpy()]


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

    def _extract_acc_or_feat(self, acc_or_feat, sample_att_func, n_samples,
                             clf_model=None, clf_indices=[], inception=None):
        clf_indices = clf_indices[:self.seq_len]

        batch_size_list = nsamples_to_batches(n_samples, self.batch_size)
        id_loss_model = IDLoss().to(self.device).eval()

        count = 0
        res_list = []
        id_losses_list = []
        id_losses_each_edit_list = []
        for bs in tqdm(batch_size_list):
            # sample ys
            ys = sample_att_func()
            ys = torch.tensor([ys]).repeat(bs, 1).to(self.device)

            imgs = []
            id_losses = []
            id_losses_each_edit = []

            z_init = z_anchor = torch.FloatTensor(ys.size(0), self.latent_dim).normal_(0, 1).to(self.device)
            z_init_cloned = z_init.clone()
            x_init_cloned = self.ccf.generate_images(self.ccf.g(z_init_cloned), seq_indices=list(range(self.seq_len)),
                                                     update_ws_prev=True)
            imgs.append(x_init_cloned)  # store images before editing

            for i in range(self.seq_len):

                seq_indices = self.get_seq_indices(i)
                x_init = self.ccf.generate_images(self.ccf.g(z_init), seq_indices=seq_indices).detach()
                img, sample_time, z_sampled = self._sample_batch(self.sampler, ys[:, seq_indices], z_init,
                                                                 z_anchor=z_anchor, seq_indices=seq_indices,
                                                                 is_return_z=True, reweight=self.reweight,
                                                                 dis_temp=self.dis_temp)

                imgs.append(img)
                attr_vals = '_'.join([f"{attr_val:.3f}" for attr_val in ys[0][:i + 1].cpu().numpy()])
                print(f'class {attr_vals}, sampling time: {sample_time}')

                id_loss = id_loss_model(x_anchor=x_init_cloned, x=img).detach()
                print(f'id loss: {id_loss.mean().item():.3f}, [check!] z_init_cloned: {z_init_cloned[:5, 0].cpu().numpy()}')
                id_losses.append(id_loss)

                id_loss_each_edit = id_loss_model(x_anchor=x_init, x=img).detach()
                print(f'id loss in each edit: {id_loss_each_edit.mean().item():.3f}')
                id_losses_each_edit.append(id_loss_each_edit)

                z_anchor = z_sampled
                if self.update_z_init and z_sampled is not None:
                    print('Updating z_init...')
                    z_init = z_sampled

            res_batches = []
            for i, img in enumerate(imgs):

                if acc_or_feat == 'acc':
                    acc_batch = classify_img(img, ys, clf_model, clf_indices)
                    res_batches.append(acc_batch)

                else:  # acc_or_feat == 'feat'
                    with torch.no_grad():
                        feat = inception(img)[0].view(img.shape[0], -1)
                    res_batches.append(feat)

            count += 1
            res_list.append(res_batches)
            id_losses_list.append(id_losses)
            id_losses_each_edit_list.append(id_losses_each_edit)

        accs_or_feats_list = [torch.cat(accs_or_feats, 0) for accs_or_feats in zip(*res_list)]
        assert accs_or_feats_list[0].size(0) == n_samples

        # calculate id loss
        id_loss_all_edits = [torch.cat(id_losses_per_edit, 0).cpu().numpy() for id_losses_per_edit
                             in zip(*id_losses_list)]  # len of edits
        for i, id_losses in enumerate(id_loss_all_edits):
            id_mean = np.mean(id_losses)
            print(f"[seq-{i + 1}] id_loss_mean: {id_mean:.3f}")

        id_loss_all_each_edit_edits = [torch.cat(id_losses_per_edit, 0).cpu().numpy() for id_losses_per_edit
                                       in zip(*id_losses_each_edit_list)]  # len of edits
        for i, id_losses_each_edit in enumerate(id_loss_all_each_edit_edits):
            id_each_edit_mean = np.mean(id_losses_each_edit)
            print(f"[seq-{i + 1}] id_loss_each_edit_mean: {id_each_edit_mean:.3f}")

        return [accs_or_feats.cpu().numpy() for accs_or_feats in accs_or_feats_list]


# ----------------------------------------------------------------------------

def nsamples_to_batches(n_samples, batch_size):
    n_batch = n_samples // batch_size
    resid = n_samples - (n_batch * batch_size)
    res = [resid] if resid > 0 else []
    batch_size_list = [batch_size] * n_batch + res
    return batch_size_list


def classify_img(img, ys, clf_model, clf_indices):
    batch, channel, height, width = img.shape
    if height > 256:
        factor = height // 256
        img = img.reshape(
            batch, channel, height // factor, factor, width // factor, factor
        )
        img = img.mean([3, 5])
    assert img.shape[-1] == 256
    img = functional.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)

    with torch.no_grad():
        logits_list = clf_model.classify(img)
        logits_list = [logits_list[j] for j in clf_indices]

    acc_list = []
    for k, logits in enumerate(logits_list):
        acc = utils.get_acc(logits, ys[:, k])
        acc_list.append(acc)

    acc_batch = torch.stack(acc_list, dim=1)

    return acc_batch


def nclasses_to_ys(n_classes_list, lo, hi, discrete_num, trim_num):
    ys_list = list(product(*[range(n_classes) if n_classes > 1 else list(np.linspace(lo, hi, num=discrete_num))
                             for n_classes in n_classes_list]))
    if 0 < trim_num < len(ys_list):
        ys_list = random.sample(ys_list, trim_num)  # without replacement
    return ys_list


def get_clf_model(device, att_names_clf='yaw-light3-smile-gender-age-glasses-pitch-haircolor-beard-bald-light0-width'):
    n_classes_list = get_n_classes_list(att_names_clf)
    load_path = os.path.join('./pretrained/metrics/', f'ckpt_{att_names_clf}_rew0.pt')

    att_names_clf_dict = {att_name: i for i, att_name in enumerate(att_names_clf.split('-'))}

    def att_names_to_indices(att_names):
        att_name_list = get_att_name_list(att_names, for_model=False)
        return [att_names_clf_dict[att_name] for att_name in att_name_list]

    # build clf model
    ffhq_clf = utils.FFHQ_Classifier(n_classes_list=n_classes_list)
    print(f"loading model from {load_path}")
    ckpt_dict = torch.load(load_path)
    ffhq_clf.load_state_dict(ckpt_dict["model_state_dict"])
    ffhq_clf = ffhq_clf.to(device)

    return ffhq_clf, att_names_to_indices


def get_attributes_sampler(att_names, att_source, seq_len=-1, root_path='./dataset_styleflow', discrete_num=9):
    att_names = get_att_name_list(att_names, for_model=False)
    n_classes_list = get_n_classes_list(att_names)
    if seq_len == -1:
        seq_len = len(n_classes_list)
    att_names = att_names[:seq_len]
    n_classes_list = n_classes_list[:seq_len]

    if att_source == 'training':  # get ngram dict from training data
        ngram_dict = get_ngram_dict(root_path, att_names)
        ys_list = list(ngram_dict.keys())
        weight_list = list(ngram_dict.values())

    elif att_source == 'uniform':  # discrete uniformly the continuous value in [lo, hi]
        lo, hi = 0.2, 0.8
        ys_list = list(product(*[range(n_classes) if n_classes > 1 else list(np.linspace(lo, hi, num=discrete_num))
                                 for n_classes in n_classes_list]))
        weight_list = [1 for _ in ys_list]

    else:
        raise NotImplementedError

    assert len(ys_list[0]) == seq_len

    def sample_att_func():
        ys = random.choices(ys_list, weights=weight_list, k=1)[0]
        return ys

    return sample_att_func


# ----------------------------------------------------------------------------


def test_clf_acc(ccf, att_names, dataset, root_path, batch_size, device):
    att_names = get_att_name_list(att_names, for_model=False)

    if dataset == "ffhq_latent_train":
        dset = FFHQLatentDataset(root_path, att_names, train=True)
        clf_func = ccf.classify_x
    elif dataset == "ffhq_latent_test":
        dset = FFHQLatentDataset(root_path, att_names, train=False)
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
