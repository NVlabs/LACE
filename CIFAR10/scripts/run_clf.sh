#!/usr/bin/env bash

# To train a classifier on cifar10_[z, w]
for x in w z; do

  python train_classifier.py --lr .0001 --up_dim 128 --optimizer adam \
    --x_space cifar10_${x} --batch_size 512 --n_epochs 50 \
    --save_dir exp_out/classifiers --warmup_iters 1000 --seed 123

  # copy ckpt to the pretrained folder
  mkdir -p pretrained/classifiers/latent/dense_embedder_${x}
  cp exp_out/classifiers/cifar10_${x}_128_bs512_seed123_/best_valid_ckpt.pt pretrained/classifiers/latent/dense_embedder_${x}/

done
