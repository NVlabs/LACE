# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for LACE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import json
import os
import argparse
import random

import numpy as np
import torch

import utils
from afhq_data import get_att_name_list
from sampling import _sample_q_dict, CCF
from eval_sampler import test_clf_acc, ConditionalSampling, SequentialEditing

LOAD_PATHS = lambda att_name, threshold: f'./pretrained/dense_embedder_w_{threshold}/best_valid_ckpt_{att_name}.pt'


def main(args):
    tag = f'{args.att_names}_{args.seq_len}/{args.sample_method}_bs{args.batch_size}' \
          f'_trunc{args.truncation}_regz{args.reg_z}_zinit{int(args.use_z_init)}_th{args.threshold}' \
          f'_reglog{args.reg_logits}_distmp{args.dis_temp}_expr{args.expr}_{args.seq_method}' \
          f'_subsel{int(args.subset_selection)}_rw{int(args.reweight)}_sd{args.seed}'
    tag += f'_atol{args.atol}_rtol{args.rtol}' if args.sample_method == 'sample_q_ode' \
        else f'_sdeN{args.sde_N}_csteps{args.correct_nsteps}_snr{args.target_snr}' if args.sample_method == 'sample_q_vpsde' \
        else f'_nsteps{args.n_steps}_ldlr{args.sgld_lr}_ldstd{args.sgld_std}'

    tag += args.tag
    save_path = os.path.join(args.save_dir, tag)
    os.makedirs(save_path, exist_ok=True)

    with open(f'{save_path}/params.txt', 'w') as f:
        json.dump(args.__dict__, f)

    logger = utils.Logger(file_name=f'{save_path}/log.txt', file_mode="a+", should_flush=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    random.seed(args.trim_seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ccf = CCF(att_names=args.att_names, expr=args.expr, latent_dim=args.latent_dim, truncation=args.truncation,
              subset_selection=args.subset_selection)

    # load em up
    att_names = get_att_name_list(args.att_names, for_model=True)
    for i, att_name in enumerate(att_names):
        load_path = LOAD_PATHS(att_name, args.threshold)
        print(f"loading model from {load_path}")

        classifier_ckpt_dict = torch.load(load_path)
        ccf.f[i].load_state_dict(classifier_ckpt_dict["state_dict"])
        ccf.f[i].to(device)

    ccf = ccf.to(device)

    print('ccf #params: {}'.format(utils.compute_n_params(ccf)))
    for i, f_i in enumerate(ccf.f):
        print('ccf.f[{}] #params: {}'.format(i, utils.compute_n_params(f_i)))
    print('ccf.G #params: {}'.format(utils.compute_n_params(ccf.G)))

    sample_q = _sample_q_dict[args.sample_method]
    ode_kwargs = {'atol': args.atol, 'rtol': args.rtol, 'method': args.method,
                  'use_adjoint': args.use_adjoint}
    ld_kwargs = {'latent_dim': args.latent_dim, 'sgld_lr': args.sgld_lr,
                 'sgld_std': args.sgld_std, 'n_steps': args.n_steps}
    sde_kwargs = {'N': args.sde_N, 'correct_nsteps': args.correct_nsteps,
                  'target_snr': args.target_snr}

    # cond_sampling and seq_editing
    cond_sampling = ConditionalSampling(args.att_names, sample_q, args.batch_size, args.latent_dim, ccf,
                                        device, args.root_path, save_path, ode_kwargs, ld_kwargs, sde_kwargs,
                                        every_n_plot=args.every_n_plot, every_n_print=args.every_n_print,
                                        use_z_init=args.use_z_init, dis_temp=args.dis_temp)

    seq_editing = SequentialEditing(args.att_names, sample_q, args.batch_size, args.latent_dim, ccf,
                                    device, args.root_path, save_path, ode_kwargs, ld_kwargs, sde_kwargs,
                                    every_n_plot=args.every_n_plot, every_n_print=args.every_n_print,
                                    use_z_init=args.use_z_init, update_z_init=True, reg_z=args.reg_z,
                                    reg_id=args.reg_id, reg_logits=args.reg_logits, seq_len=args.seq_len,
                                    seq_method=args.seq_method, reweight=args.reweight, dis_temp=args.dis_temp)

    # evaluate the latent classifier performance
    if args.test_clf:
        print('starting test_clf...')
        ccf.eval()
        with torch.no_grad():
            test_clf_acc(ccf, args.att_names, args.dataset, args.root_path, args.batch_size, device)
        ccf.train()

    # conditional sampling (uniform)
    if args.cond:
        print('starting cond...')
        cond_sampling.get_samples(discrete_num=args.discrete_num, trim_num=args.trim_num)

    # conditional sampling (use-specified)
    if args.cond_use_case:
        print('starting cond_use_case...')
        cond_sampling.get_samples(att_vals=args.att_vals, discrete_num=args.discrete_num, trim_num=args.trim_num)

    # sequential editing (uniform)
    if args.seq_edit:
        print('starting seq_edit...')
        seq_editing.get_samples(discrete_num=args.discrete_num, trim_num=args.trim_num)

    # sequential editing (use-specified)
    if args.seq_edit_use_case:
        print('starting seq_edit_use_case...')
        seq_editing.get_samples(att_vals=args.att_vals, att_vals_init=args.att_vals_init,
                                discrete_num=args.discrete_num, trim_num=args.trim_num,
                                zinit_seed=args.zinit_seed)

    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training Classifiers is All You Need")
    parser.add_argument("--root_path", type=str, default="./ada-afhqcat-latents")
    parser.add_argument("--att_names", type=str, default="gender")
    parser.add_argument("--att_vals", type=str, default="")
    parser.add_argument("--att_vals_init", type=str, default="")
    parser.add_argument("--sample_method", type=str, default='ode',
                        choices=['sample_q_ode', 'sample_q_sgld', 'sample_q_vpsde'],
                        help="sampling method for conditional generation")
    parser.add_argument("--dataset", default="afhqcat_latent_train", type=str,
                        choices=["afhqcat_latent_train", "afhqcat_latent_test"],
                        help="Dataset to use when running test_clf for classification accuracy")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--trim_seed", type=int, default=0)
    parser.add_argument("--zinit_seed", type=int, default=1)
    parser.add_argument("--discrete_num", type=int, default=3)
    parser.add_argument("--trim_num", type=int, default=0)
    parser.add_argument("--seq_method", type=str, default='Method1', choices=['Method1', 'Method2'])
    parser.add_argument("--expr", type=str, default="")

    parser.add_argument("--truncation", type=float, default=1, help='1: no truncation, (0, 1): truncation')
    parser.add_argument("--dis_temp", type=float, default=1, help='(0.1, 10): adjust discrete control')
    parser.add_argument("--tag", type=str, default='')

    # EBM specific
    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--sgld_lr", type=float, default=1.0)
    parser.add_argument("--sgld_std", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--reg_z", type=float, default=0)
    parser.add_argument("--use_z_init", action="store_true")
    parser.add_argument("--update_z_init", action="store_true")
    parser.add_argument("--reg_id", type=float, default=0)
    parser.add_argument("--reg_logits", type=float, default=0)
    parser.add_argument("--seq_len", type=int, default=0)
    parser.add_argument("--reg_w", type=float, default=0)
    # ODE
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--use_adjoint", type=bool, default=True)
    parser.add_argument("--method", type=str, default='dopri5')

    # SDE
    parser.add_argument("--sde_N", type=int, default=1000)
    parser.add_argument("--correct_nsteps", type=int, default=2)
    parser.add_argument("--target_snr", type=float, default=0.16)

    # logging + evaluation
    parser.add_argument("--save_dir", type=str, default='save')
    parser.add_argument("--every_n_plot", type=int, default=5)
    parser.add_argument("--every_n_print", type=int, default=5)
    parser.add_argument("--test_clf", action="store_true")
    parser.add_argument("--cond", action="store_true")
    parser.add_argument("--seq_edit", action="store_true")
    parser.add_argument("--cond_use_case", action="store_true")
    parser.add_argument("--seq_edit_use_case", action="store_true")
    parser.add_argument("--subset_selection", action="store_true")
    parser.add_argument("--reweight", action="store_true")

    parser.add_argument('--threshold', type=float, default=0.3, help='threshold for clip classification')

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = parser.parse_args()
    main(args)
