# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for LACE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import numpy as np
import torch.nn as nn


def get_norm(n_filters, norm):
    if norm is None:
        return nn.Identity()
    elif norm == "batch":
        return nn.BatchNorm2d(n_filters, momentum=0.9)
    elif norm == "instance":
        return nn.InstanceNorm2d(n_filters, affine=True)
    elif norm == "layer":
        return nn.GroupNorm(1, n_filters)


class DenseEmbedder(nn.Module):
    """Supposed to map small-scale features (e.g. labels) to some given latent dim"""
    def __init__(self, input_dim, up_dim, depth=4, num_classes=10, given_dims=None, norm=None):
        super().__init__()
        self.net = nn.ModuleList()
        if given_dims is not None:
            assert given_dims[0] == input_dim
            assert given_dims[-1] == up_dim
            dims = given_dims
        else:
            dims = np.linspace(input_dim, up_dim, depth).astype(int)

        for l in range(len(dims)-1):
            self.net.append(nn.Conv2d(dims[l], dims[l + 1], 1))

            self.net.append(get_norm(dims[l + 1], norm))
            self.net.append(nn.LeakyReLU(0.2))

        self.last_dim = up_dim
        self.linear = nn.Linear(up_dim, num_classes)

        print('Using DenseEmbedder...')
        print(f'{norm} norm')

    def forward(self, x):
        if x.ndim == 2:
            x = x[:, :, None, None]

        for layer in self.net:
            x = layer(x)

        out = x.squeeze(-1).squeeze(-1)
        out = self.linear(out)
        return out


