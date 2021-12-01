#!/usr/bin/env bash

# ode (cifar10_w)
for x in w; do
  for seed in 125; do

    # sampling
    python main.py --x_space cifar10_$x --sample_method sample_q_ode --cifar10_pretrained dense_embedder_$x \
      --n_steps 100 --sgld_lr 1e-2 --sgld_std 1e-2 --batch_size 64 --save_dir exp_out/cond_ode --dataset cifar_test --seed $seed \
      --atol 1e-3 --rtol 1e-3 --use_adjoint True --method dopri5 --inception pretrained/metrics/inception_cifar10.pkl \
      --test_clf \
      --cond --every_n_plot 100

  done
done

