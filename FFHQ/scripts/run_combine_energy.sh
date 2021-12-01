#!/usr/bin/env bash

# logical operators (ODE) - (AND, OR, NEG)
for att_names_vals in "glasses_yaw 1._0.5"; do
  set -- $att_names_vals
  for expr in '0+1' '0*1' '0+(-1)'; do

    # user-specified sampling
    python main.py --att_names $1 --sample_method sample_q_ode --save_dir exp_out/combine_energy \
      --n_steps 100 --sgld_lr 1e-2 --sgld_std 1e-2 --batch_size 16 --dataset ffhq_latent_test --seed 129 \
      --atol 1e-3 --rtol 1e-3 --use_adjoint True --method dopri5 --truncation 0.7 --discrete_num 3 --test_clf \
      --expr $expr \
      --cond_use_case --att_vals $2 --every_n_plot 100 --every_n_print 40 --trim_num 3 --trim_seed 0

  done
done

# logical operators (ODE) - (RECURSIVE)
for att_names_vals in "glasses_yaw_glasses_yaw 1._0.5_0._0.5"; do
  set -- $att_names_vals
  for expr in '(0+1)*(2+(-3))'; do

    # user-specified sampling
    python main.py --att_names $1 --sample_method sample_q_ode --save_dir exp_out/combine_energy \
      --n_steps 100 --sgld_lr 1e-2 --sgld_std 1e-2 --batch_size 16 --dataset ffhq_latent_test --seed 129 \
      --atol 1e-3 --rtol 1e-3 --use_adjoint True --method dopri5 --truncation 0.7 --discrete_num 3 --test_clf \
      --expr $expr \
      --cond_use_case --att_vals $2 --every_n_plot 100 --every_n_print 40 --trim_num 3 --trim_seed 0

  done
done
