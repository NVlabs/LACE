#!/usr/bin/env bash

# train each attribute classifier in sequence
for att_name in yaw smile gender age glasses light3 pitch haircolor beard bald light0 width; do

  python train_classifier.py --lr .0001 --up_dim 128 --optimizer adam \
    --att_names $att_name --batch_size 128 --n_epochs 50 \
    --save_dir exp_out/classifiers --warmup_iters 1000 --seed 123

  # copy ckpt to the pretrained folder
  mkdir -p pretrained/dense_embedder_w
  cp exp_out/classifiers/${att_name}_bs128_seed123_/best_valid_ckpt_${att_name}.pt pretrained/dense_embedder_w/

done
