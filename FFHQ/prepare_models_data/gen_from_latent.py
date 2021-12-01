# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for LACE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import pickle
import numpy as np
import PIL.Image
import sys
sys.path.append('../')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch

from stylegan2 import Generator


def load_stylegan2(latent, ckpt_path):
    ckpt = f'{ckpt_path}/stylegan2_pt/stylegan2-ffhq-config-f.pt'

    g_ema = Generator(1024, 512, 8, channel_multiplier=2).to('cuda')
    checkpoint = torch.load(ckpt)
    g_ema.load_state_dict(checkpoint["g_ema"])
    g_ema.eval()

    sample, _ = g_ema([latent], randomize_noise=False, input_is_latent=True)

    return sample


def gen_all_images(ws, batch_size, root_path, ckpt_path):
    save_dir = f'{root_path}/images'
    os.makedirs(save_dir, exist_ok=True)

    print(f'ws shape: {ws.shape}')
    n_batches = ws.shape[0] // batch_size + 1

    print(n_batches)

    for i in range(n_batches):
        w_batch = ws[i * batch_size: (i + 1) * batch_size]
        if w_batch.shape[0] == 0:
            break
        sample = load_stylegan2(w_batch, ckpt_path)

        img = (sample.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img_np = img.cpu().numpy()
        for idx in range(w_batch.shape[0]):
            PIL.Image.fromarray(img_np[idx], 'RGB').save(f'{save_dir}/{(i * batch_size + idx):05d}.png')

        if i % 200 == 0:
            print(i)


if __name__ == '__main__':

    root_path = '../dataset_styleflow'
    ckpt_path = '../pretrained'
    # dataset_styleflow
    all_latents = pickle.load(open(f"{root_path}/all_latents.pickle", "rb"))
    ws_np = np.array(all_latents['Latent'])
    ws = torch.tensor(ws_np).squeeze().to('cuda')

    print('starting to generate all images...')
    batch_size = 8
    gen_all_images(ws, batch_size, root_path, ckpt_path)