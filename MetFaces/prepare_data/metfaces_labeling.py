# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for LACE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
import os
import pickle
import sys
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data as data
from tqdm import tqdm

from metrics.img_classifier import MobileNet
from metfaces_data import MetFacesDataset, get_att_name_list, get_n_classes_list


class F(nn.Module):
    def __init__(self, n_classes_list=[2]):
        super(F, self).__init__()
        self.backbone = MobileNet()
        self.n_classes_list = n_classes_list

        self.last_dim = 1024
        self.heads = nn.Linear(self.last_dim, sum(n_classes_list))

        print('Initialized the ffhq image classifier!')

    def forward(self, x):
        x = self.backbone.features(x)
        x = x.view(x.size(0), -1)
        logits = self.heads(x)
        logits_list = list(torch.split(logits, self.n_classes_list, dim=1))
        return logits_list


def get_clf_model(device, load_dir, att_names_clf='yaw-light3-smile-gender-age-glasses-pitch-haircolor-beard-bald-light0-width'):
    n_classes_list = get_n_classes_list(att_names_clf)
    load_path = os.path.join(load_dir, f'ckpt_{att_names_clf}_rew0.pt')

    att_names_clf_dict = {att_name: i for i, att_name in enumerate(att_names_clf.split('-'))}

    def att_names_to_indices(att_names):
        att_name_list = get_att_name_list(att_names, for_model=False)
        return [att_names_clf_dict[att_name] for att_name in att_name_list]

    # build clf model
    ffhq_clf = F(n_classes_list=n_classes_list)
    assert load_path is not None
    print(f"loading model from {load_path}")
    ckpt_dict = torch.load(load_path)
    ffhq_clf.load_state_dict(ckpt_dict["model_state_dict"])
    ffhq_clf = ffhq_clf.to(device)

    return ffhq_clf, att_names_to_indices


def labeling_gen(gen_loader, model, use_cuda):

    # switch to evaluate mode
    model.eval()

    preds, idxes = [], []
    for batch_idx, (img_idxes, inputs) in tqdm(enumerate(gen_loader)):

        if use_cuda:
            inputs = inputs.cuda()

        # compute output (as logits_list)
        with torch.no_grad():
            logits_list = model(inputs)

        pred_list = []
        for i, logits in enumerate(logits_list):
            n_classes = logits.size(1)
            if n_classes > 1:
                _, pred = logits.topk(k=1, dim=1, largest=True, sorted=True)
            else:
                pred = logits
            pred_list.append(pred)

        preds.append(torch.cat(pred_list, dim=1))
        idxes.append(img_idxes)

    return torch.cat(preds, dim=0).squeeze(), torch.cat(idxes).squeeze()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Label metfaces by using ffhq classifier")
    parser.add_argument("--root_path", type=str, default="../ada-metfaces-latents")
    parser.add_argument("--load_dir", type=str, default="../../FFHQ/pretrained/metrics")
    parser.add_argument('--test_batch', default=16, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--out_dir', type=str, default='../ada-metfaces-latents', help='Where to save predictions')
    args = parser.parse_args()

    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = torch.cuda.is_available()
    print(f'use cuda: {use_cuda}')
    device = torch.device('cuda' if use_cuda else 'cpu')

    metfaces_dst = MetFacesDataset(root_path=args.root_path, res=224)
    gen_loader = data.DataLoader(metfaces_dst, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    clf_model, att_names_to_indices = get_clf_model(device, load_dir=args.load_dir)

    print('\nLabeling generated metfaces_dst images only')
    preds, idxes = labeling_gen(gen_loader, clf_model, use_cuda)
    print(f'preds shape: {preds.shape}')
    print(f'idxes shape: {idxes.shape}')

    os.makedirs(args.out_dir, exist_ok=True)

    pred_dict = {'pred': [preds.cpu().numpy()]}
    idx_dict = {'index': [idxes.cpu().numpy()]}
    with open(f'{args.out_dir}/pred.pickle', "wb") as f:
        pickle.dump(pred_dict, f)
    with open(f'{args.out_dir}/index.pickle', "wb") as f:
        pickle.dump(idx_dict, f)
