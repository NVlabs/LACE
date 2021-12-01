#!/usr/bin/env bash

# generate 10k w, images
python generate_batch.py --outdir=../ada-metfaces-latents --seeds=0-199 --bs 50 \
  --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
