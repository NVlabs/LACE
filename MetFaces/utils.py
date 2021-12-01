# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for LACE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import sys
from typing import Any

import torch
import torch.nn as nn


def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


class Logger(object):
    """
    Redirect stderr to stdout, optionally print stdout to a file,
    and optionally force flushing on both stdout and the file.
    """

    def __init__(self, file_name: str = None, file_mode: str = "w", should_flush: bool = True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def write(self, text: str) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if len(text) == 0: # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()


def get_loss(logits, y, device='cuda', reduce=False):
    assert y.ndim == 1 and logits.ndim == 2, (y.ndim, logits.ndim)
    n_classes = logits.size(1)
    eff_idxes = (y != -1)
    logits, y = logits[eff_idxes], y[eff_idxes]

    if n_classes > 1:  # discrete attribute
        y = y.long()
        weight = torch.tensor([1 for _ in range(n_classes)]).float().to(device)
        reduction = 'mean' if reduce else 'none'
        loss = nn.CrossEntropyLoss(reduction=reduction, weight=weight)(logits, y)
    else:  # continuous attribute
        assert n_classes == 1, n_classes
        y = y.float()
        weight = torch.tensor([1 for _ in range(y.size(0))]).float().to(device)
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
        return correct.detach(), num_c
    else:
        return correct.detach()


def logical_comb(s: str, es: list):
    stack = [0, '+']
    val = 0.
    s += '$'
    for c in s:

        if c.isnumeric():
            val = es[int(c)]
        elif c in ['+', '-', '*', ')', '$']:

            symbol = stack.pop()
            if symbol == '+':
                stack[-1] += val
            elif symbol == '-':
                alpha = torch.clamp(0.1 / torch.abs(val.detach().clone()), min=0, max=1.)
                stack[-1] -= val * alpha
            else:
                tl, tr, cof = 1.0, 1.0, 20.0  # these values work well
                stack[-1] = torch.log(torch.exp(stack[-1] * tl) * cof + torch.exp(val * tr))

            if c == ')':
                val = stack.pop()
                stack.pop()
            elif c in ['+', '-', '*']:
                stack.append(c)
                val = 0.

        elif c == '(':
            stack.append(c)
            stack.extend([0, '+'])

    return stack[-1]


def get_z_inits(num, batch_size, latent_dim, device):
    z_0 = torch.FloatTensor(batch_size, latent_dim).normal_(0, 1).to(device)
    z_1 = torch.FloatTensor(batch_size, latent_dim).normal_(0, 1).to(device)

    def rescale_z(z):
        z_mean_norm = (torch.linalg.norm(z_0, dim=1, keepdim=True) + torch.linalg.norm(z_1, dim=1, keepdim=True)) / 2
        assert z_mean_norm.shape[0] == batch_size
        return z / torch.linalg.norm(z, dim=1, keepdim=True) * z_mean_norm

    z_inits = [z_0]
    for i in range(1, num):
        z_k = z_0 + (i / num) * (z_1 - z_0)
        z_inits.append(rescale_z(z_k))
    z_inits.append(z_1)

    return z_inits


# heuristics from the styleflow paper: https://github.com/RameenAbdal/StyleFlow
subset_dict = {
    'light0': list(range(7, 12)),
    'light3': list(range(7, 12)),
    'smile': list(range(4, 6)),
    'yaw': list(range(0, 4)),
    'pitch': list(range(0, 4)),
    'age': list(range(4, 8)),
    'gender': list(range(0, 8)),
    'glasses': list(range(0, 6)),
    'bald': list(range(0, 6)),
    'beard': list(range(5, 10)),
}


def subset_from_att_names(att_names_list):
    set_all = list(range(18))
    if len(att_names_list) == 0:  # no subset selection
        return []

    subset_sel = []
    for att_name in att_names_list:
        subset_sel += subset_dict.get(att_name, set_all)
    subset_nonsel = [i for i in set_all if i not in set(subset_sel)]

    return subset_nonsel

