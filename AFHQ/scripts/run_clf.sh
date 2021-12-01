#!/usr/bin/env bash

for threshold in 0.3; do
  mkdir -p pretrained/dense_embedder_w_$threshold

  for att_name in breeds haircolor moods age; do

    python train_classifier.py --lr .0001 --up_dim 128 --optimizer adam \
      --att_names $att_name --batch_size 128 --n_epochs 50 \
      --save_dir exp_out/classifiers --warmup_iters 1000 --seed 123 \
      --threshold $threshold

    # copy ckpt to pretrained
    cp exp_out/classifiers/${att_name}_th${threshold}_bs128_seed123_/best_valid_ckpt_${att_name}.pt pretrained/dense_embedder_w_$threshold/

  done
done
