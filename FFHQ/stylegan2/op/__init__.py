# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from a file in the rosinality/stylegan2-pytorch repository
# which was released under the MIT License.
#
# Source:
# https://github.com/rosinality/stylegan2-pytorch/blob/master/op/__init__.py
#
# The license for the original version of this file can be
# found in https://github.com/rosinality/stylegan2-pytorch/blob/master/LICENSE.
# The modifications to this file are subject to the same MIT License.
# ---------------------------------------------------------------

from .fused_act import FusedLeakyReLU, fused_leaky_relu
from .upfirdn2d import upfirdn2d
