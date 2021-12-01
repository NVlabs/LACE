# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for LACE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import argparse
import numpy as np
import json
from tqdm import tqdm
import sys
sys.path.append('../')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from metrics.img_classifier import MobileNet
from ffhq_data import FFHQGenDataset, get_att_name_list, get_n_classes_list

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


# manually designed reweighting functions (continuous)
bald_reweight_func = lambda x: torch.log(1 + 1000 * x)
age_reweight_func = lambda x: 1 / torch.exp(-7 * torch.abs(x - 0.5)) if x > 0.2 else torch.tensor(8.166)


# manually designed reweighting functions (discrete)
def normalize_func(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return torch.max(x) / x


att_dis_weights = {'smile': normalize_func([1939, 6520]),
                   'gender': normalize_func([4375, 5619]),
                   'beard': normalize_func([8800, 1194]),
                   'glasses': normalize_func([8448, 1546]),
                   'haircolor': normalize_func([6282, 459, 1871, 900, 60])}

att_cont_weights = {'age': age_reweight_func,
                    'bald': bald_reweight_func}


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
    def __init__(self, n_classes_list=[2]):
        super(F, self).__init__()
        self.backbone = MobileNet()
        self.n_classes_list = n_classes_list

        self.last_dim = 1024
        self.heads = nn.Linear(self.last_dim, sum(n_classes_list))

    def classify(self, x):
        x = self.backbone.features(x)
        x = x.view(x.size(0), -1)
        logits = self.heads(x)
        logits_list = list(torch.split(logits, self.n_classes_list, dim=1))
        return logits_list


def get_data(args):
    # get all training inds
    full_train = FFHQGenDataset(args.root_path, args.att_names, train=True)
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
        FFHQGenDataset(args.root_path, args.att_names, train=True),
        inds=train_inds)
    dset_valid = DataSubset(
        FFHQGenDataset(args.root_path, args.att_names, train=True),
        inds=valid_inds)
    dload_train = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    dload_valid = DataLoader(dset_valid, batch_size=64, shuffle=False, num_workers=4, drop_last=False)

    dset_test = FFHQGenDataset(args.root_path, args.att_names, train=False)
    dload_test = DataLoader(dset_test, batch_size=64, shuffle=False, num_workers=4, drop_last=False)
    return dload_train, dload_valid, dload_test


def get_loss(logits, y, att_name=None, device=torch.device('cuda'), reduce=False):
    assert y.ndim == 1 and logits.ndim == 2, (y.ndim, logits.ndim)
    n_classes = logits.size(1)
    eff_idxes = (y != -1)
    logits, y = logits[eff_idxes], y[eff_idxes]

    if n_classes > 1:  # discrete attribute
        y = y.long()
        weight = torch.tensor([1 for _ in range(n_classes)]).float().to(device)
        if att_name is not None:  # add reweighting
            weight = att_dis_weights[att_name].to(device)
        reduction = 'mean' if reduce else 'none'
        loss = nn.CrossEntropyLoss(reduction=reduction, weight=weight)(logits, y)
    else:  # continuous attribute
        assert n_classes == 1, n_classes
        y = y.float()
        weight = torch.tensor([1 for _ in range(y.size(0))]).float().to(device)
        if att_name is not None:  # add reweighting
            weight = torch.tensor(list(map(att_cont_weights[att_name], y))).float().to(device)
        loss = torch.linalg.norm(logits - y[:, None], dim=1) ** 2 * 0.5 * weight
        if reduce:
            loss = loss.mean()

    assert loss.ndim == 1 if not reduce else loss.ndim == 0
    return loss


def get_acc(logits, y, c=None):
    assert y.ndim == 1 and logits.ndim == 2, (y.ndim, logits.ndim)
    n_classes = logits.size(1)
    eff_idxes = (y != -1)
    logits, y = logits[eff_idxes], y[eff_idxes]

    if n_classes > 1:  # discrete attribute
        y = y.long()
        if c is None:
            correct = (logits.max(1)[1] == y).float()
        else:
            correct = ((logits.max(1)[1] == y) * (y == c)).float()
            num_c = (y == c).sum()
    else:  # continuous attribute
        assert n_classes == 1, n_classes
        y = y.float()
        correct = 1 - torch.abs(logits.squeeze() - y)

    assert correct.ndim == 1
    if n_classes > 1 and c is not None:
        return correct, num_c
    else:
        return correct


def eval_classification(f, dload, att_names, reweight, device):
    corrects_list, losses_list = [[] for _ in att_names], [[] for _ in att_names]
    n_classes_list = get_n_classes_list(att_names)
    corrects_per_cls_list = [[[] for _ in range(n_classes)] for n_classes in n_classes_list]
    nums_per_cls_list = [[0 for _ in range(n_classes)] for n_classes in n_classes_list]

    for x_p_d, ys in dload:
        x_p_d, ys = x_p_d.to(device), ys.to(device)
        logits_list = f.classify(x_p_d)
        for i, logits in enumerate(logits_list):
            att_name_or_none = att_names[i] if reweight else None
            loss = get_loss(logits, ys[:, i], att_name=att_name_or_none, device=device).detach().cpu().numpy()
            losses_list[i].extend(loss)

            correct = get_acc(logits, ys[:, i]).cpu().numpy()
            corrects_list[i].extend(correct)

            if logits.size(1) > 1:  # output acc of each class for discrete attribute
                for c in range(logits.size(1)):
                    acc_c, nums_c = get_acc(logits, ys[:, i], c=c)
                    corrects_per_cls_list[i][c].extend(acc_c.cpu().numpy())
                    nums_per_cls_list[i][c] += nums_c.cpu().numpy()

    loss_per_att, correct_per_att, correct_per_att_per_cls = [], [], []
    for losses, corrects, corrects_per_cls, nums_per_cls in zip(
            losses_list, corrects_list, corrects_per_cls_list, nums_per_cls_list):
        loss_per_att.append(np.mean(losses))
        correct_per_att.append(np.mean(corrects))
        correct_cls_list = [(np.sum(corrects_c) / num_c, num_c)
                            for corrects_c, num_c in zip(corrects_per_cls, nums_per_cls)]

        correct_per_att_per_cls.append(correct_cls_list)

    correct = np.mean(correct_per_att)
    loss = np.mean(loss_per_att)
    return correct, correct_per_att, correct_per_att_per_cls, loss


def checkpoint(f, tag, save_path, device):
    f.cpu()
    ckpt_dict = {
        "model_state_dict": f.state_dict(),
    }
    torch.save(ckpt_dict, os.path.join(save_path, tag))
    f.to(device)


def main(args):
    save_path = os.path.join(args.save_dir, f'{args.att_names}_bs{args.batch_size}_seed{args.seed}_{args.tag}')

    os.makedirs(save_path, exist_ok=True)
    with open(f'{save_path}/params_{args.att_names}_rew{int(args.reweight)}.txt', 'w') as f:
        json.dump(args.__dict__, f)
    if args.print_to_log:
        sys.stdout = open(f'{save_path}/log_{args.att_names}_rew{int(args.reweight)}.txt', 'w')

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # datasets
    dload_train, dload_valid, dload_test = get_data(args)

    att_names = get_att_name_list(args.att_names, for_model=False)
    n_classes_list = get_n_classes_list(args.att_names)

    # build latent model
    f = F(n_classes_list=n_classes_list)
    if args.load_path is not None:
        print(f"loading model from {args.load_path}")
        ckpt_dict = torch.load(args.load_path)
        f.backbone.load_state_dict(ckpt_dict["model_state_dict"])
    f = f.to(device)

    print(f'Reweight: {args.reweight}')

    # optimizer
    params = f.parameters()
    if args.optimizer == "adam":
        optim = torch.optim.Adam(params, lr=args.lr, betas=(.9, .999), weight_decay=args.weight_decay)
    else:
        optim = torch.optim.SGD(params, lr=args.lr, momentum=.9, weight_decay=args.weight_decay)

    best_valid_acc = 0.0
    cur_iter = 0

    for epoch in range(args.n_epochs):
        if epoch in args.decay_epochs:
            for param_group in optim.param_groups:
                new_lr = param_group['lr'] * args.decay_rate
                param_group['lr'] = new_lr
            print("Decaying lr to {}".format(new_lr))

        for i, (x, ys) in tqdm(enumerate(dload_train)):
            if cur_iter <= args.warmup_iters:
                lr = args.lr * cur_iter / float(args.warmup_iters)
                for param_group in optim.param_groups:
                    param_group['lr'] = lr

            x, ys = x.to(device), ys.to(device)

            logits_list = f.classify(x)
            loss = 0.
            for i, logits in enumerate(logits_list):
                att_name_or_none = att_names[i] if args.reweight else None
                loss += get_loss(logits, ys[:, i], reduce=True, att_name=att_name_or_none, device=device)

            if cur_iter % args.print_every == 0:
                for i, logits in enumerate(logits_list):
                    correct = get_acc(logits, ys[:, i]).mean()

                    print('{} {}:{:>d} loss={:>14.9f}, correct={:>14.9f}'.format(att_names[i], epoch,
                                                                                 cur_iter,
                                                                                 loss.item(),
                                                                                 correct.item()))

            optim.zero_grad()
            loss.backward()
            optim.step()
            cur_iter += 1

        if epoch % args.ckpt_every == 0:
            checkpoint(f, f'ckpt_{epoch}_{args.att_names}_rew{int(args.reweight)}.pt', save_path, device)

        if epoch % args.eval_every == 0:
            f.eval()
            with torch.no_grad():
                # validation set
                correct, correct_per_att, correct_per_att_per_cls, loss \
                    = eval_classification(f, dload_valid, att_names, args.reweight, device)
                print("Epoch {}: Valid Loss {}, Valid Acc {} | {}: {}, {}".format(epoch, loss, correct, att_names,
                                                                                  correct_per_att,
                                                                                  correct_per_att_per_cls))
                if correct > best_valid_acc:
                    best_valid_acc = correct
                    print("Best Valid!: {}".format(correct))
                    checkpoint(f, f"best_valid_ckpt_{args.att_names}_rew{int(args.reweight)}.pt", save_path, device)
                # test set
                correct, correct_per_att, correct_per_att_per_cls, loss \
                    = eval_classification(f, dload_test, att_names, args.reweight, device)
                print("Epoch {}: Test Loss {}, Test Acc {} | {}: {}, {}".format(epoch, loss, correct, att_names,
                                                                                correct_per_att,
                                                                                correct_per_att_per_cls))
            f.train()
        checkpoint(f, f"last_ckpt_{args.att_names}_rew{int(args.reweight)}.pt", save_path, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("training ffhq image classifier")
    parser.add_argument("--root_path", type=str, default="../dataset_styleflow")
    parser.add_argument("--att_names", type=str, default="gender")
    # optimization
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay_epochs", nargs="+", type=int, default=[160, 180],
                        help="decay learning rate by decay_rate at these epochs")
    parser.add_argument("--decay_rate", type=float, default=.3,
                        help="learning rate decay multiplier")
    parser.add_argument("--labels_per_class", type=int, default=-1,
                        help="number of labeled examples per class, if zero then use all labels")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--reweight", type=int, default=1, choices=[0, 1])
    parser.add_argument("--warmup_iters", type=int, default=-1,
                        help="number of iters to linearly increase learning rate, if -1 then no warmmup")

    # logging + evaluation
    parser.add_argument("--save_dir", type=str, default='./experiment')
    parser.add_argument("--ckpt_every", type=int, default=10, help="Epochs between checkpoint save")
    parser.add_argument("--eval_every", type=int, default=1, help="Epochs between evaluation")
    parser.add_argument("--print_every", type=int, default=100, help="Iterations between print")
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--print_to_log", action="store_true", help="If true, directs std-out to log file")
    parser.add_argument("--n_valid", type=int, default=500)
    # tag
    parser.add_argument("--tag", type=str, default='')

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    args = parser.parse_args()
    main(args)
