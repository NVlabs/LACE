#!/usr/bin/env bash

for att_name in yaw-light3-smile-gender-age-glasses-pitch-haircolor-beard-bald-light0-width; do

  python train_ffhq_classifiers.py --lr .0001 --optimizer adam \
    --load_path ./model_checkpoint.pth \
    --att_names $att_name --batch_size 128 --n_epochs 50 --reweight 0 \
    --save_dir out --warmup_iters 1000 --seed 124 \
    --print_to_log

  # copy ckpt to the pretrained folder
  mkdir -p ../pretrained/metrics
  cp out/${att_name}_bs128_seed123_/best_valid_ckpt_${att_name}_rew0.pt ../pretrained/metrics/

done
