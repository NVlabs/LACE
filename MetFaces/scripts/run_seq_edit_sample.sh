#!/usr/bin/env bash

# sequential editing (ODE)
for att_names in yaw_beard_haircolor_age_bald_gender_light3_pitch_light0_width_smile_glasses; do
  for seq_len in 3; do
    for seed in 125; do

      # uniform sampling
      python main.py --att_names $att_names --sample_method sample_q_ode --save_dir exp_out/seq_edit \
        --n_steps 100 --sgld_lr 1e-2 --sgld_std 1e-2 --batch_size 16 --dataset metfaces_latent_test --seed $seed \
        --atol 1e-3 --rtol 1e-3 --use_adjoint True --method dopri5 --truncation 0.5 --discrete_num 3 --test_clf \
        --update_z_init --reg_z 0.04 --reg_id 0 --reg_logits 0.01 --seq_len $seq_len \
        --seq_edit --every_n_plot 200 --every_n_print 40 --use_z_init --trim_num 5 --trim_seed $seed

    done
  done
done
