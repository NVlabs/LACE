#!/usr/bin/env bash

# sequential editing (ODE)
for att_names in yaw_smile_gender_age_glasses_light3_pitch_haircolor_beard_bald_light0_width; do
  for seq_len in 5; do
    for seed in 125; do

      # ACC + uniform sampling
      python main.py --att_names $att_names --sample_method sample_q_ode --save_dir exp_out/seq_edit \
        --n_steps 100 --sgld_lr 1e-2 --sgld_std 1e-2 --batch_size 16 --dataset ffhq_latent_test --seed $seed \
        --atol 1e-3 --rtol 1e-3 --use_adjoint True --method dopri5 --truncation 0.5 --discrete_num 3 --test_clf \
        --att_source uniform \
        --eval_seq_edit_acc \
        --update_z_init --reg_z 0.04 --reg_id 0 --reg_logits 0.01 --seq_len $seq_len \
        --seq_edit --every_n_plot 200 --every_n_print 40 --use_z_init --trim_num 3 --trim_seed $seed

      # FID
      python main.py --att_names $att_names --sample_method sample_q_ode --save_dir exp_out/seq_edit \
        --n_steps 100 --sgld_lr 1e-2 --sgld_std 1e-2 --batch_size 16 --dataset ffhq_latent_test --seed $seed \
        --atol 1e-3 --rtol 1e-3 --use_adjoint True --method dopri5 --truncation 0.5 --discrete_num 3 --test_clf \
        --att_source training \
        --eval_seq_edit_fid --inception_pkl ./pretrained/metrics/inception_ffhq_gen_1k.pkl \
        --update_z_init --reg_z 0.04 --reg_id 0 --reg_logits 0.01 --seq_len $seq_len \

    done
  done
done

