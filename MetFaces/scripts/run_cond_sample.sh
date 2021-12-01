#!/usr/bin/env bash

# conditional generation (ODE)
for att_names in yaw haircolor gender_yaw; do
  for seed in 125; do

  # uniform sampling
  python main.py --att_names $att_names --sample_method sample_q_ode --save_dir exp_out/cond_ode \
    --n_steps 100 --sgld_lr 1e-2 --sgld_std 1e-2 --batch_size 16 --dataset metfaces_latent_test --seed $seed \
    --atol 1e-3 --rtol 1e-3 --use_adjoint True --method dopri5 --truncation 0.7 --discrete_num 3 --test_clf \
    --cond --every_n_plot 1000 --every_n_print 200 --trim_num 10 --trim_seed $seed

done
done
