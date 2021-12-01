#!/usr/bin/env bash

# conditional generation (ODE)
for att_names_vals in "haircolor_age 1_0" "breeds 6" "breeds_haircolor_moods 15_2_0"; do
  set -- $att_names_vals
  for seed in 125; do
    for threshold in 0.3; do
      for dis_temp in 0.7; do

        # user-specified sampling
        python main.py --att_names $1 --sample_method sample_q_ode --save_dir exp_out/cond_ode \
          --n_steps 100 --sgld_lr 1e-2 --sgld_std 1e-2 --batch_size 16 --dataset afhqcat_latent_test --seed $seed \
          --atol 1e-3 --rtol 1e-3 --use_adjoint True --method dopri5 --truncation 0.7 --discrete_num 3 \
          --cond_use_case --att_vals $2 --every_n_plot 200 --every_n_print 40 --trim_num 5 --trim_seed 0 \
          --threshold $threshold \
          --dis_temp $dis_temp \

      done
    done
  done
done
