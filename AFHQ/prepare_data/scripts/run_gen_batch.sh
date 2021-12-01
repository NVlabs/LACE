#!/usr/bin/env bash

# generate 10k w, images
python generate_batch.py --outdir=../ada-afhqcat-latents --seeds=0-199 --bs 50 \
  --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/afhqcat.pkl
