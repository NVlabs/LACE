#!/usr/bin/env bash

# sequential editing (ODE)
for att_names in yaw_smile_gender_age_glasses_light3_pitch_haircolor_beard_bald_light0_width; do
  for seq_len_att_vals in "5 0.75_1._1._0.75_1. 0.50_0._0._0.30_0."; do
    set -- $seq_len_att_vals
    for seed in 125; do

      # user-specified sampling
      python main.py --att_names $att_names --sample_method sample_q_ode --save_dir exp_out/seq_edit \
        --n_steps 100 --sgld_lr 1e-2 --sgld_std 1e-2 --batch_size 16 --dataset ffhq_latent_test --seed $seed \
        --atol 1e-3 --rtol 1e-3 --use_adjoint True --method dopri5 --truncation 0.5 --discrete_num 3 --test_clf \
        --update_z_init --reg_z 0.04 --reg_id 0 --reg_logits 0.01 --seq_len $1 --att_vals_init $3 \
        --seq_edit_use_case --att_vals $2 --every_n_plot 200 --every_n_print 40 --use_z_init --trim_num 3 --trim_seed $seed

    done
  done
done
