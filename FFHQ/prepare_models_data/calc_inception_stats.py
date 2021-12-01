# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from a file in the rosinality/stylegan2-pytorch repository
# which was released under the MIT License.
#
# Source:
# https://github.com/rosinality/stylegan2-pytorch/blob/master/calc_inception.py
#
# The license for the original version of this file can be
# found in https://github.com/rosinality/stylegan2-pytorch/blob/master/LICENSE.
# The modifications to this file are subject to the same MIT License.
# ---------------------------------------------------------------

import argparse
import pickle
import os
import sys
sys.path.append('../')

import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from metrics.calc_inception import load_patched_inception_v3
from ffhq_data import FFHQGenDataset


@torch.no_grad()
def extract_features(loader, inception, device):
    feature_list = []

    for img, _ in tqdm(loader):
        img = img.to(device)
        feature = inception(img)[0].view(img.shape[0], -1)
        feature_list.append(feature.to("cpu"))

    features = torch.cat(feature_list, 0)

    return features


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(
        description="Calculate Inception v3 features for datasets"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1024,
        help="image sizes used for embedding calculation",
    )
    parser.add_argument(
        "--batch", default=100, type=int, help="batch size for inception networks"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=1000,
        help="number of samples used for embedding calculation",
    )
    parser.add_argument(
        "--flip", action="store_true", help="apply random flipping to real images"
    )
    parser.add_argument("--dataset", default='ffhq_gen', help="dataset type")
    parser.add_argument("--data_path", default='../dataset_styleflow', help="path to dataset file")
    parser.add_argument("--print_to_log", action="store_true")

    args = parser.parse_args()

    if args.print_to_log:
        log_path = 'incept_log'
        os.makedirs(log_path, exist_ok=True)
        sys.stdout = open(f'{log_path}/log.txt', 'w')

    inception = load_patched_inception_v3()
    inception = nn.DataParallel(inception).eval().to(device)

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5 if args.flip else 0),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    if args.dataset == 'ffhq_gen':  # res: 1024
        dset = FFHQGenDataset(args.data_path, 'light0', train=True)
    else:
        raise NotImplementedError
    loader = DataLoader(dset, batch_size=args.batch, num_workers=4)

    features = extract_features(loader, inception, device).numpy()
    features = features[: args.n_sample]

    print(f"extracted {features.shape[0]} features")

    mean = np.mean(features, 0)
    cov = np.cov(features, rowvar=False)

    save_path = '../pretrained/metrics'
    os.makedirs(save_path, exist_ok=True)
    with open(f"{save_path}/inception_{args.dataset}.pkl", "wb") as f:
        pickle.dump({"mean": mean, "cov": cov, "size": args.size, "path": args.data_path}, f)
