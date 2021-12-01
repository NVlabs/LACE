# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# This file is modified from StyleGAN2-ADA.
# See https://github.com/NVlabs/stylegan2-ada/blob/main/LICENSE.txt for the license file.

import os
import re
from typing import List, Optional
import sys
sys.path.append('../')

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import pickle

import legacy


# ----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    vals = s.split(',')
    return [int(x) for x in vals]


# ----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, show_default=True, help='Class label (unconditional if not specified)')
@click.option('--bs', 'batch_size', type=int, default=1, help='Batch size for each sampling')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const',
              show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def generate_images(
        ctx: click.Context,
        network_pkl: str,
        seeds: Optional[List[int]],
        truncation_psi: float,
        noise_mode: str,
        outdir: str,
        batch_size: Optional[int],
        class_idx: Optional[int],
        projected_w: Optional[str]
):
    """Generate 10k images using pretrained network pickle.
    python generate_batch.py --outdir=out --seeds=0-199 --bs 300
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig11b-cifar10/cifar10u-cifar-ada-best-fid.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Synthesize the result of a W projection.
    if projected_w is not None:
        if seeds is not None:
            print('warn: --seeds is ignored when using --projected-w')
        print(f'Generating images from projected W "{projected_w}"')
        ws = pickle.load(open(projected_w, 'rb'))['ws_latent'][0][:5]
        ws = torch.tensor(ws, device=device)  # pylint: disable=not-callable
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        for idx, w in enumerate(ws):
            img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/proj{idx:02d}.png')
        return

    if seeds is None:
        ctx.fail('--seeds option is required when not using --projected-w')

    # Labels.
    label = torch.zeros([batch_size, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')

    # Generate images (#seeds * batch_size).
    z_dict = {'z_latent': []}
    ws_dict = {'ws_latent': []}

    imagedir = f'{outdir}/images'
    os.makedirs(imagedir, exist_ok=True)

    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(batch_size, G.z_dim)).to(device)
        ws = G.mapping(z, label, truncation_psi=truncation_psi)
        img = G.synthesis(ws, noise_mode=noise_mode)

        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img_np = img.cpu().numpy()
        for i in range(img_np.shape[0]):
            save_id = seed_idx * batch_size + i
            PIL.Image.fromarray(img_np[i], 'RGB').save(f'{imagedir}/seed{save_id:05d}.png')
        z_dict['z_latent'].append(z)
        ws_dict['ws_latent'].append(ws)

    z_dict['z_latent'] = [torch.cat(z_dict['z_latent']).cpu().numpy()]
    ws_dict['ws_latent'] = [torch.cat(ws_dict['ws_latent']).cpu().numpy()]
    with open(f'{outdir}/z_latent.pickle',"wb") as f:
        pickle.dump(z_dict, f)
    with open(f'{outdir}/ws_latent.pickle',"wb") as f:
        pickle.dump(ws_dict, f)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    generate_images()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------