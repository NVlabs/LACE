#!/usr/bin/env bash

# ld (cifar10_w)
for x in w; do
  for seed in 125; do

    # eval acc
    python main.py --x_space cifar10_$x --sample_method sample_q_sgld --cifar10_pretrained dense_embedder_$x \
      --n_steps 100 --sgld_lr 1e-2 --sgld_std 1e-2 --batch_size 400 --save_dir exp_out/cond_ld --dataset cifar_test --seed $seed \
      --atol 1e-3 --rtol 1e-3 --use_adjoint True --method dopri5 --inception pretrained/metrics/inception_cifar10.pkl \
      --test_clf \
      --eval_cond_acc

    # eval fid
    python main.py --x_space cifar10_$x --sample_method sample_q_sgld --cifar10_pretrained dense_embedder_$x \
      --n_steps 100 --sgld_lr 1e-2 --sgld_std 1e-2 --batch_size 400 --save_dir exp_out/cond_ld --dataset cifar_test --seed $seed \
      --atol 1e-3 --rtol 1e-3 --use_adjoint True --method dopri5 --inception pretrained/metrics/inception_cifar10.pkl \
      --test_clf \
      --eval_cond_fid

  done
done
