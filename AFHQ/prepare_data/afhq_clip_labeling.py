# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for LACE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
import os
import time
import pickle
import sys
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data as data
from tqdm import tqdm

import clip
from afhq_data import AFHQCatDataset, cat_attr_classes_dict, cat_attr_sentences_dict


class CLIP_Model(nn.Module):
    def __init__(self, device):
        super(CLIP_Model, self).__init__()
        self.model, self.preprocess = clip.load('ViT-B/16', device)
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=16)

        self.clip_sentence_fn = lambda attr_name, attr_val: cat_attr_sentences_dict[attr_name].replace('xxx', attr_val)
        self.clip_attr_list = list(cat_attr_sentences_dict.keys())

        self.n_classes_list = [len(val) for val in cat_attr_classes_dict.values()]

        print('Initialized the clip model!')

    def forward(self, image, threshold=0.3):
        img_inputs = self.avg_pool(self.upsample(image))

        if not isinstance(threshold, list):
            threshold = [threshold] * len(self.clip_attr_list)

        classes_list = []
        for i, attr_name in enumerate(self.clip_attr_list):
            attr_classes = cat_attr_classes_dict[attr_name]
            text_inputs = torch.cat([clip.tokenize(self.clip_sentence_fn(attr_name, attr_val))
                                     for attr_val in attr_classes]).to(device)

            # Calculate features
            with torch.no_grad():
                image_features = self.model.encode_image(img_inputs)
                text_features = self.model.encode_text(text_inputs)

            # Pick the top 1 most similar labels for the image
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            probs, class_ids = similarity.topk(1, dim=-1)

            # Use threshold
            class_ids[probs < threshold[i]] = -1  # bs * 1
            classes_list.append(class_ids)

        preds = torch.cat(classes_list, dim=-1)  # bs * n_att
        assert preds.shape[-1] == len(self.clip_attr_list)
        return preds


def labeling_gen(gen_loader, clip_model, use_cuda, threshold=0.3):

    preds, idxes = [], []

    end = time.time()
    for batch_idx, (img_idxes, inputs) in tqdm(enumerate(gen_loader)):

        if use_cuda:
            inputs = inputs.cuda()

        # compute output (as logits_list)
        with torch.no_grad():
            preds_per_batch = clip_model(inputs, threshold=threshold)

        preds.append(preds_per_batch)
        idxes.append(img_idxes)

        if batch_idx % 10 == 0:
            print(f'batch idx: {batch_idx}')

    return torch.cat(preds, dim=0).squeeze(), torch.cat(idxes).squeeze()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Label afhq-cat by using clip")
    parser.add_argument("--root_path", type=str, default="../ada-afhqcat-latents")
    parser.add_argument('--test_batch', default=32, type=int, metavar='N',
                        help='test batch size')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--outdir', type=str, default='out', help='Where to save the output predictions')
    parser.add_argument('--threshold', type=float, default=0.3, help='threshold for clip classification')
    args = parser.parse_args()

    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = torch.cuda.is_available()
    print(f'use cuda: {use_cuda}')
    device = torch.device('cuda' if use_cuda else 'cpu')

    afhqcat_dst = AFHQCatDataset(root_path=args.root_path)
    gen_loader = data.DataLoader(afhqcat_dst, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    clip_model = CLIP_Model(device)

    print('\nLabeling generated afhqcat_dst images only')
    preds, idxes = labeling_gen(gen_loader, clip_model, use_cuda, threshold=args.threshold)
    print(f'preds shape: {preds.shape}')
    print(f'idxes shape: {idxes.shape}')

    # statistics on preds
    for i in range(preds.shape[1]):
        print(f'{clip_model.clip_attr_list[i]}: '
              f'{[(preds[:, i] == k).sum().item() for k in range(clip_model.n_classes_list[i])]}')
        print(f'{clip_model.clip_attr_list[i]}: '
              f'{(preds[:, i] == -1).sum().item()}')

    os.makedirs(args.outdir, exist_ok=True)

    pred_dict = {f'pred{args.threshold:.1f}': [preds.cpu().numpy()]}
    idx_dict = {'index': [idxes.cpu().numpy()]}
    pred_fn = f'{args.outdir}/pred{args.threshold:.1f}.pickle'
    with open(pred_fn, "wb") as f:
        pickle.dump(pred_dict, f)
    idx_fn = f'{args.outdir}/index.pickle'
    with open(idx_fn, "wb") as f:
        pickle.dump(idx_dict, f)

    # cp
    os.system(f'cp {pred_fn} {args.root_path}')
    os.system(f'cp {idx_fn} {args.root_path}')
