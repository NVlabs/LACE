#!/usr/bin/env bash

# generate 60k w, images
python generate_batch.py --outdir=../ada-cifar10-latents --seeds=0-199 --bs 300 \
  --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig11b-cifar10/cifar10u-cifar-ada-best-fid.pkl
