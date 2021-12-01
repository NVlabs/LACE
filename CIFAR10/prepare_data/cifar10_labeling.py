# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for LACE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
import os
import random
import pickle
import sys
sys.path.append('../')

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import models

from cifar10_data import Cifar10GenDataset


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))


parser = argparse.ArgumentParser(description='CIFAR10 Labeling')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--test-batch', default=64, type=int, metavar='N',
                    help='test batchsize')
# Optimization options
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: densenet)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck '
                         '(default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-l', '--labeling_gen', action='store_true', help='labeling the generated images')
parser.add_argument('--outdir', type=str, default='../ada-cifar10-latents', help='Where to save the output predictions')
parser.add_argument('--root_path', type=str, default='../ada-cifar10-latents', help='Where to load images')
# Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10', 'Dataset can only be cifar10.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)


def main():

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    num_classes = 10

    # cifar-10-gen Data
    cifar10_gen_dst = Cifar10GenDataset(root_path=args.root_path)
    gen_loader = data.DataLoader(cifar10_gen_dst, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('densenet'):
        model = models.DenseNet(
            num_classes=num_classes,
            depth=args.depth,
            growthRate=args.growthRate,
            compressionRate=args.compressionRate,
            dropRate=args.drop,
        )
    elif args.arch.startswith('wrn'):
        model = models.WideResNet(
            num_classes=num_classes,
            depth=args.depth,
            widen_factor=args.widen_factor,
            dropRate=args.drop,
        )
    else:
        model = models.DenseNet(num_classes=num_classes)

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
    args.checkpoint = os.path.dirname(args.resume)
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])

    if args.labeling_gen:
        print('\nLabeling generated Cifar-10 images only')
        preds, idxes = labeling_gen(gen_loader, model, use_cuda)

        os.makedirs(args.outdir, exist_ok=True)

        pred_dict = {'pred': [preds.cpu().numpy()]}
        idx_dict = {'index': [idxes.cpu().numpy()]}
        with open(f'{args.outdir}/pred.pickle', "wb") as f:
            pickle.dump(pred_dict, f)
        with open(f'{args.outdir}/index.pickle', "wb") as f:
            pickle.dump(idx_dict, f)
        return


def labeling_gen(gen_loader, model, use_cuda):

    # switch to evaluate mode
    model.eval()

    preds, idxes = [], []

    for batch_idx, (img_idxes, inputs) in enumerate(gen_loader):

        if use_cuda:
            inputs = inputs.cuda()

        # compute output
        outputs = model(inputs)

        _, pred = outputs.topk(k=1, dim=1, largest=True, sorted=True)
        preds.append(pred)
        idxes.append(img_idxes)

        if batch_idx % 100 == 0:
            print(f'batch_idx: {batch_idx}')

    return torch.cat(preds).squeeze(), torch.cat(idxes).squeeze()


if __name__ == '__main__':
    main()
