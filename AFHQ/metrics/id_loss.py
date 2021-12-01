# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from a file in the InsightFace_Pytorch library
# which was released under the MIT License.
#
# Source:
# https://github.com/TreB1eN/InsightFace_Pytorch
#
# The license for the original version of this file can be
# found in https://github.com/TreB1eN/InsightFace_Pytorch/blob/master/LICENSE.
# The modifications to this file are subject to the same MIT License.
# ---------------------------------------------------------------

import torch
from torch import nn

from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Dropout, Sequential, Module
from .TreB1eN import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE, l2_norm


class Backbone(Module):
    def __init__(self, input_size, num_layers, mode='ir', drop_ratio=0.4, affine=True):
        super(Backbone, self).__init__()
        assert input_size in [112, 224], "input_size should be 112 or 224"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        if input_size == 112:
            self.output_layer = Sequential(BatchNorm2d(512),
                                           Dropout(drop_ratio),
                                           Flatten(),
                                           Linear(512 * 7 * 7, 512),
                                           BatchNorm1d(512, affine=affine))
        else:
            self.output_layer = Sequential(BatchNorm2d(512),
                                           Dropout(drop_ratio),
                                           Flatten(),
                                           Linear(512 * 14 * 14, 512),
                                           BatchNorm1d(512, affine=affine))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return l2_norm(x)


model_paths = {'ir_se50': './pretrained/metrics/model_ir_se50.pth'}


class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(model_paths['ir_se50']))
        self.face_pool_pre = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        if x.shape[-1] > 256:
            x = self.face_pool_pre(x)
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, x_anchor, x):
        n_samples = x.shape[0]
        x_feats = self.extract_feats(x)
        x_anchor_feats = self.extract_feats(x_anchor)
        x_anchor_feats = x_anchor_feats.detach()

        assert x_anchor_feats.ndim == x_feats.ndim == 2, (x_feats.ndim, x_anchor_feats.ndim)
        loss = 1 - torch.einsum('bi,bi->b', x_feats, x_anchor_feats)

        return loss
