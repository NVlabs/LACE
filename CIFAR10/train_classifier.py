# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for LACE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import argparse
import time
import numpy as np
import json
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import utils
import latent_model
from cifar10_data import Cifar10LatentDataset


class DataSubset(Dataset):
    def __init__(self, base_dataset, inds=None, size=-1):
        self.base_dataset = base_dataset
        if inds is None:
            inds = np.random.choice(list(range(len(base_dataset))), size, replace=False)
        self.inds = inds

    def __getitem__(self, index):
        base_ind = self.inds[index]
        return self.base_dataset[base_ind]

    def __len__(self):
        return len(self.inds)


class F(nn.Module):
    def __init__(self, input_dim, up_dim=128, norm=None, n_classes=10):
        super(F, self).__init__()
        self.f = latent_model.DenseEmbedder(input_dim, up_dim, norm=norm, num_classes=n_classes)

    def classify(self, x):
        logits = self.f(x).squeeze()
        return logits


def get_data(args):
    # get all training inds
    full_train = Cifar10LatentDataset(args.root_path, args.x_space, train=True)
    all_inds = list(range(len(full_train)))
    # set seed
    np.random.seed(1234)
    # shuffle
    np.random.shuffle(all_inds)
    # separate out validation set
    if args.n_valid is not None:
        valid_inds, train_inds = all_inds[:args.n_valid], all_inds[args.n_valid:]
    else:
        valid_inds, train_inds = [], all_inds
    train_inds = np.array(train_inds)

    dset_train = DataSubset(
        Cifar10LatentDataset(args.root_path, args.x_space, train=True),
        inds=train_inds)
    dset_valid = DataSubset(
        Cifar10LatentDataset(args.root_path, args.x_space, train=True),
        inds=valid_inds)
    dload_train = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    dload_valid = DataLoader(dset_valid, batch_size=100, shuffle=False, num_workers=4, drop_last=False)

    dset_test = Cifar10LatentDataset(args.root_path, args.x_space, train=False)
    dload_test = DataLoader(dset_test, batch_size=100, shuffle=False, num_workers=4, drop_last=False)
    return dload_train, dload_valid, dload_test


def eval_classification(f, dload, device):
    corrects, losses = [], []
    for x_p_d, y_p_d in dload:
        x_p_d, y_p_d = x_p_d.to(device), y_p_d.to(device)
        logits = f.classify(x_p_d)
        loss = nn.CrossEntropyLoss(reduce=False)(logits, y_p_d).cpu().numpy()
        losses.extend(loss)
        correct = (logits.max(1)[1] == y_p_d).float().cpu().numpy()
        corrects.extend(correct)
    loss = np.mean(losses)
    correct = np.mean(corrects)
    return correct, loss


def checkpoint(f, tag, save_path, device):
    f.cpu()
    ckpt_dict = {
        "model_state_dict": f.state_dict(),
        "state_dict": f.f.state_dict(),
    }
    torch.save(ckpt_dict, os.path.join(save_path, tag))
    f.to(device)


def main(args):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    latent_dim = 512

    save_path = os.path.join(args.save_dir, f'{args.x_space}_{args.up_dim}_bs{args.batch_size}_'
                                            f'seed{args.seed}_{args.tag}')

    os.makedirs(save_path, exist_ok=True)
    with open(f'{save_path}/params.txt', 'w') as f:
        json.dump(args.__dict__, f)

    logger = utils.Logger(file_name=f'{save_path}/log.txt', file_mode="a+", should_flush=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # datasets
    dload_train, dload_valid, dload_test = get_data(args)

    # build latent model
    f = F(latent_dim, up_dim=args.up_dim, norm=args.norm)
    if args.load_path is not None:
        print(f"loading model from {args.load_path}")
        ckpt_dict = torch.load(args.load_path)
        f.load_state_dict(ckpt_dict["model_state_dict"])
    f = f.to(device)

    # optimizer
    params = f.parameters()
    if args.optimizer == "adam":
        optim = torch.optim.Adam(params, lr=args.lr, betas=(.9, .999), weight_decay=args.weight_decay)
    else:
        optim = torch.optim.SGD(params, lr=args.lr, momentum=.9, weight_decay=args.weight_decay)

    best_valid_acc = 0.0
    cur_iter = 0

    start_time = time.time()
    for epoch in range(args.n_epochs):
        if epoch in args.decay_epochs:
            for param_group in optim.param_groups:
                new_lr = param_group['lr'] * args.decay_rate
                param_group['lr'] = new_lr
            print("Decaying lr to {}".format(new_lr))

        for i, (x, y) in tqdm(enumerate(dload_train)):
            if cur_iter <= args.warmup_iters:
                lr = args.lr * cur_iter / float(args.warmup_iters)
                for param_group in optim.param_groups:
                    param_group['lr'] = lr

            x, y = x.to(device), y.to(device)

            logits = f.classify(x)
            loss = nn.CrossEntropyLoss()(logits, y)

            if cur_iter % args.print_every == 0:
                acc = (logits.max(1)[1] == y).float().mean()
                print('P(y|x) {}:{:>d} loss={:>14.9f}, acc={:>14.9f}'.format(epoch,
                                                                             cur_iter,
                                                                             loss.item(),
                                                                             acc.item()))

            optim.zero_grad()
            loss.backward()
            optim.step()
            cur_iter += 1

        if epoch % args.ckpt_every == 0:
            checkpoint(f, f'ckpt_{epoch}.pt', save_path, device)
            print('time elapsed after {} epochs: {}'.format(epoch, time.time() - start_time))

        if epoch % args.eval_every == 0:
            f.eval()
            with torch.no_grad():
                # validation set
                correct, loss = eval_classification(f, dload_valid, device)
                print("Epoch {}: Valid Loss {}, Valid Acc {}".format(epoch, loss, correct))
                if correct > best_valid_acc:
                    best_valid_acc = correct
                    print("Best Valid!: {}".format(correct))
                    checkpoint(f, "best_valid_ckpt.pt", save_path, device)
                # test set
                correct, loss = eval_classification(f, dload_test, device)
                print("Epoch {}: Test Loss {}, Test Acc {}".format(epoch, loss, correct))
            f.train()
        checkpoint(f, "last_ckpt.pt", save_path, device)

    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train latent classifier")
    parser.add_argument("--root_path", type=str, default="./ada-cifar10-latents")
    # optimization
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay_epochs", nargs="+", type=int, default=[160, 180],
                        help="decay learning rate by decay_rate at these epochs")
    parser.add_argument("--decay_rate", type=float, default=.3,
                        help="learning rate decay multiplier")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--warmup_iters", type=int, default=-1,
                        help="number of iters to linearly increase learning rate, if -1 then no warmmup")
    # network
    parser.add_argument("--norm", type=str, default=None, choices=[None, "norm", "batch", "instance", "layer", "act"],
                        help="norm to add to weights, none works fine")
    parser.add_argument("--up_dim", type=int, default=128, help="LatentEBMnet last layer dim")
    parser.add_argument("--x_space", type=str, default='cifar10_w', choices=['cifar10_z', 'cifar10_w'],
                        help="latent input (z, w)")
    # logging + evaluation
    parser.add_argument("--save_dir", type=str, default='./experiment')
    parser.add_argument("--ckpt_every", type=int, default=10, help="Epochs between checkpoint save")
    parser.add_argument("--eval_every", type=int, default=1, help="Epochs between evaluation")
    parser.add_argument("--print_every", type=int, default=100, help="Iterations between print")
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--n_valid", type=int, default=5000)
    # tag
    parser.add_argument("--tag", type=str, default='')

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    args = parser.parse_args()
    main(args)
