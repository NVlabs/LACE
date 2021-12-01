#!/usr/bin/env bash

# conditional generation (ODE)
for att_names in glasses gender_smile_age; do
  for seed in 125; do

    # ACC + uniform sampling
    python main.py --att_names $att_names --sample_method sample_q_ode --save_dir exp_out/cond_ode \
      --n_steps 100 --sgld_lr 1e-2 --sgld_std 1e-2 --batch_size 16 --dataset ffhq_latent_test --seed $seed \
      --atol 1e-3 --rtol 1e-3 --use_adjoint True --method dopri5 --truncation 0.7 --discrete_num 3 --test_clf \
      --att_source uniform \
      --eval_cond_acc \
      --cond --every_n_plot 100 --every_n_print 40 --use_z_init --trim_num 3 --trim_seed 0

    # FID
    python main.py --att_names $att_names --sample_method sample_q_ode --save_dir exp_out/cond_ode \
      --n_steps 100 --sgld_lr 1e-2 --sgld_std 1e-2 --batch_size 16 --dataset ffhq_latent_test --seed $seed \
      --atol 1e-3 --rtol 1e-3 --use_adjoint True --method dopri5 --truncation 0.7 --discrete_num 3 --test_clf \
      --att_source training \
      --eval_cond_fid \
      --inception_pkl ./pretrained/metrics/inception_ffhq_gen_1k.pkl

  done
done
