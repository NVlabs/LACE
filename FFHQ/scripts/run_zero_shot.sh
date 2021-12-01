#!/usr/bin/env bash

# zero-shot generation (ODE)
for att_names_vals in "smile_age_glasses_beard 0._0.22_1._1." "gender_smile_glasses_age 1._0._1._0.14"; do
  set -- $att_names_vals

  # user-specified sampling
  python main.py --att_names $1 --sample_method sample_q_ode --save_dir exp_out/zero_shot \
    --n_steps 100 --sgld_lr 1e-2 --sgld_std 1e-2 --batch_size 16 --dataset ffhq_latent_test --seed 124 \
    --atol 1e-3 --rtol 1e-3 --use_adjoint True --method dopri5 --truncation 0.7 --discrete_num 3 --test_clf \
    --att_source uniform \
    --cond_use_case --att_vals $2 --every_n_plot 100 --every_n_print 40 --trim_num 3 --trim_seed 0

done
