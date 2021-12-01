# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for LACE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import json
import os
import argparse

import numpy as np
import torch

import utils
from sampling import _sample_q_dict, CCF
from eval_sampler import test_clf, ConditionalSampling


LOAD_PATHS = {
    'densenet-bc-L190-k40': './pretrained/classifiers/cifar10/densenet-bc-L190-k40/model_best.pth.tar',
    'WRN-28-10-drop': './pretrained/classifiers/cifar10/WRN-28-10-drop/model_best.pth.tar',
    'dense_embedder_z': './pretrained/classifiers/latent/dense_embedder_z/best_valid_ckpt.pt',
    'dense_embedder_w': './pretrained/classifiers/latent/dense_embedder_w/best_valid_ckpt.pt',
}


def main(args):
    tag = f'{args.sample_method}_bs{args.batch_size}_sd{args.seed}'
    tag += f'_atol{args.atol}_rtol{args.rtol}' if args.sample_method == 'sample_q_ode' \
        else f'_sdeN{args.sde_N}_csteps{args.correct_nsteps}_snr{args.target_snr}' if args.sample_method == 'sample_q_vpsde' \
        else f'_nsteps{args.n_steps}_sgld_lr{args.sgld_lr}_sgld_std{args.sgld_std}'
    save_path = os.path.join(args.save_dir, args.x_space, tag)
    os.makedirs(save_path, exist_ok=True)

    with open(f'{save_path}/params.txt', 'w') as f:
        json.dump(args.__dict__, f)

    logger = utils.Logger(file_name=f'{save_path}/log.txt', file_mode="a+", should_flush=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ccf = CCF(x_space=args.x_space, cifar10_pretrained=args.cifar10_pretrained, latent_dim=args.latent_dim)

    # load em up
    load_path = LOAD_PATHS[args.cifar10_pretrained]
    print(f"loading model from {load_path}")

    classifier_ckpt_dict = torch.load(load_path)
    ccf.f.load_state_dict(classifier_ckpt_dict["state_dict"])

    ccf = ccf.to(device)

    print('ccf #params: {}'.format(utils.compute_n_params(ccf)))
    print('ccf.f #params: {}'.format(utils.compute_n_params(ccf.f)))
    print('ccf.G #params: {}'.format(utils.compute_n_params(ccf.G)))

    sample_q = _sample_q_dict[args.sample_method]
    ode_kwargs = {'atol': args.atol, 'rtol': args.rtol, 'method': args.method, 'use_adjoint': args.use_adjoint}
    ld_kwargs = {'batch_size': args.batch_size, 'latent_dim': args.latent_dim, 'sgld_lr': args.sgld_lr,
                 'sgld_std': args.sgld_std, 'n_steps': args.n_steps}
    sde_kwargs = {'N': args.sde_N, 'correct_nsteps': args.correct_nsteps, 'target_snr': args.target_snr}

    n_classes = 10
    condSampling = ConditionalSampling(sample_q, args.batch_size, args.latent_dim, n_classes, ccf,
                                       device, save_path, ode_kwargs, ld_kwargs, sde_kwargs,
                                       every_n_plot=args.every_n_plot)

    # evaluate classifier performance
    if args.test_clf:
        print('starting test_clf...')
        ccf.eval()
        with torch.no_grad():
            test_clf(ccf, args.x_space, args.dataset, args.root_path, args.batch_size, device)
        ccf.train()

    # conditional sampling
    if args.cond:
        print('starting cond sampling...')
        condSampling.get_samples()

    # evaluate the fid score
    if args.eval_cond_fid:
        print('starting evaluating fid50k...')
        condSampling.eval_fid(inception_pkl=args.inception)

    # evaluate the cond_acc score
    if args.eval_cond_acc:
        print('starting evaluating cond_acc...')
        condSampling.eval_acc()

    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training Classifier is All You Need")
    parser.add_argument("--root_path", type=str, default="./ada-cifar10-latents")
    parser.add_argument("--x_space", type=str, default='cifar10_z',
                        choices=['cifar10_z', 'cifar10_w', 'cifar10_i'],
                        help="latent input (z, w) or generated image input")
    parser.add_argument("--sample_method", type=str, default='ode',
                        choices=['sample_q_ode', 'sample_q_sgld', 'sample_q_vpsde'],
                        help="sampling method for conditional generation")
    parser.add_argument("--cifar10_pretrained", type=str, default='densenet-bc-L190-k40',
                        choices=['densenet-bc-L190-k40', 'WRN-28-10-drop', 'dense_embedder_z', 'dense_embedder_w'],
                        help='What classifier to choose, be consistent to x_space')
    parser.add_argument("--dataset", default="cifar_test", type=str,
                        choices=["cifar_train", "cifar_test"],
                        help="Dataset to use when running test_clf for classification accuracy")
    parser.add_argument("--seed", type=int, default=123)
    # EBM specific
    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--sgld_lr", type=float, default=1.0)
    parser.add_argument("--sgld_std", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--latent_dim", type=int, default=512)
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
    parser.add_argument("--test_clf", action="store_true")
    parser.add_argument("--eval_cond_fid", action="store_true")
    parser.add_argument("--inception", type=str, default='pretrained/metrics/inception_cifar10.pkl',
                        help="path to precomputed inception embedding")
    parser.add_argument("--eval_cond_acc", action="store_true")
    parser.add_argument("--cond", action="store_true")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = parser.parse_args()
    main(args)
