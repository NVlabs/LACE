# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for LACE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch
import torch.nn as nn

from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint as odeint_normal

import dnnlib, legacy
from latent_model import DenseEmbedder
from models import DenseNet
from models import WideResNet


NETWORK_PKL = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig11b-cifar10/cifar10u-cifar-ada-best-fid.pkl'


class F(nn.Module):
    def __init__(self, x_space, cifar10_pretrained='densenet-bc-L190-k40', latent_dim=512, n_classes=10):
        super(F, self).__init__()
        self.x_space = x_space

        print('Loading networks from "%s"...' % NETWORK_PKL)
        with dnnlib.util.open_url(NETWORK_PKL) as f:
            G = legacy.load_network_pkl(f)['G_ema']  # type: ignore
        self.G = G

        if x_space == 'cifar10_i':

            if cifar10_pretrained == 'densenet-bc-L190-k40':
                self.f = DenseNet(num_classes=n_classes, depth=190, growthRate=40, compressionRate=2, dropRate=0)
            elif cifar10_pretrained == 'WRN-28-10-drop':
                self.f = WideResNet(num_classes=n_classes, depth=28, widen_factor=10, dropRate=0.3)
            else:
                raise NotImplementedError('unknown cifar10_pretrained, choices: [densenet-bc-L190-k40, WRN-28-10-drop]')

            self.f = nn.DataParallel(self.f)

            def mapping_z_to_i(z):
                label = torch.zeros([z.shape[0], G.c_dim])
                assert z.shape[-1] == G.z_dim
                return G(z, label, truncation_psi=1, noise_mode='const')

            self.g = mapping_z_to_i

        elif x_space == 'cifar10_w':
            self.f = DenseEmbedder(input_dim=latent_dim, up_dim=128, num_classes=n_classes, norm=None)

            def mapping_z_to_w(z):
                assert z.shape[-1] == G.z_dim
                label = torch.zeros([z.shape[0], self.G.c_dim])
                ws = G.mapping(z, label, truncation_psi=1)
                return ws[:, 0, :]

            self.g = mapping_z_to_w

        elif x_space == 'cifar10_z':
            self.f = DenseEmbedder(input_dim=latent_dim, up_dim=128, num_classes=n_classes, norm=None)
            self.g = nn.Identity()

        else:
            raise NotImplementedError('unknown x_space, choices: [cifar10_i, cifar10_w, cifar10_z]')

    def classify(self, z):
        logits = self.f(self.g(z)).squeeze()
        return logits

    def classify_x(self, x):
        logits = self.f(x).squeeze()
        return logits

    def generate_images(self, g_z_sampled, is_detached=True):
        """Generate images from g_z_sampled or dataset"""

        # Synthesize the result from i
        if self.x_space == 'cifar10_i':
            img = g_z_sampled
            return img.detach() if is_detached else img

        # Synthesize the result from w
        if self.x_space == 'cifar10_w':
            ws = g_z_sampled
            if ws.ndim == 2:
                ws = ws.unsqueeze(1).repeat([1, 8, 1])  # bs x 8 x 512
            assert ws.shape[1:] == (self.G.num_ws, self.G.w_dim)
            img = self.G.synthesis(ws, noise_mode='const')
            return img.detach() if is_detached else img

        # Synthesize the result from z
        if self.x_space == 'cifar10_z':
            z = g_z_sampled
            label = torch.zeros([z.shape[0], self.G.c_dim])
            assert z.shape[-1] == self.G.z_dim
            ws = self.G.mapping(z, label, truncation_psi=1)
            img = self.G.synthesis(ws, noise_mode='const')
            return img.detach() if is_detached else img

        return None


class CCF(F):
    def __init__(self, x_space, cifar10_pretrained='densenet-bc-L190-k40', latent_dim=512):
        super(CCF, self).__init__(x_space, cifar10_pretrained, latent_dim=latent_dim)

        self.x_space = x_space
        print(f'Working in the x_space: {x_space}')

    def get_cond_energy(self, z, y):
        logits = self.classify(z)
        energy_output = torch.gather(logits, 1, y[:, None]).squeeze() - logits.logsumexp(1)
        return energy_output

    def forward(self, z, y):
        energy_output = self.get_cond_energy(z, y) - torch.linalg.norm(z, dim=1) ** 2 * 0.5
        return energy_output


# ----------------------------------------------------------------------------

_sample_q_dict = dict()  # name => fn


def register_sample_q(fn):
    assert callable(fn)
    _sample_q_dict[fn.__name__] = fn
    return fn


@register_sample_q
def sample_q_sgld(ccf, y, device=torch.device('cuda'), save_path=None, plot=None, every_n_plot=5, **kwargs):
    """sampling in the z space"""
    ccf.eval()

    latent_dim = kwargs['latent_dim']
    sgld_lr = kwargs['sgld_lr']
    sgld_std = kwargs['sgld_std']
    n_steps = kwargs['n_steps']

    # generate initial samples
    init_sample = torch.randn(y.size(0), latent_dim).to(device)
    x_k = torch.autograd.Variable(init_sample, requires_grad=True)

    # sgld
    for k in range(n_steps):

        if save_path is not None and k % every_n_plot == 0:
            g_z_sampled = ccf.g(x_k.detach())
            x_sampled = ccf.generate_images(g_z_sampled)
            plot('{}/samples_class{}_nsteps{}.png'.format(save_path, y[0].item(), k), x_sampled)

        energy_neg = ccf(x_k, y=y)
        f_prime = torch.autograd.grad(energy_neg.sum(), [x_k])[0]
        x_k.data += sgld_lr * f_prime + sgld_std * torch.randn_like(x_k)

    ccf.train()
    final_samples = x_k.detach()

    return final_samples


class VPODE(nn.Module):
    def __init__(self, ccf, y, beta_min=0.1, beta_max=20, T=1.0, save_path=None, plot=None, every_n_plot=5):
        super().__init__()
        self.ccf = ccf
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.T = T
        self.y = y
        self.save_path = save_path
        self.n_evals = 0
        self.every_n_plot = every_n_plot
        self.plot = plot

    def forward(self, t_k, states):
        z = states[0]

        if self.save_path is not None and self.n_evals % self.every_n_plot == 0:
            g_z_sampled = self.ccf.g(z.detach())
            x_sampled = self.ccf.generate_images(g_z_sampled)
            self.plot(f'{self.save_path}/samples_cls{self.y[0].item()}_nsteps{self.n_evals:03d}_tk{t_k}.png',
                      x_sampled)

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)

            beta_t = self.beta_0 + t_k * (self.beta_1 - self.beta_0)
            cond_energy_neg = self.ccf.get_cond_energy(z, self.y)
            cond_f_prime = torch.autograd.grad(cond_energy_neg.sum(), [z])[0]
            dz_dt = -0.5 * beta_t * cond_f_prime

        self.n_evals += 1

        return dz_dt,


@register_sample_q
def sample_q_ode(ccf, y, device=torch.device('cuda'), save_path=None, plot=None, every_n_plot=5, **kwargs):
    """sampling in the z space"""
    ccf.eval()

    latent_dim = kwargs['latent_dim']
    atol = kwargs['atol']
    rtol = kwargs['rtol']
    method = kwargs['method']
    use_adjoint = kwargs['use_adjoint']

    # generate initial samples
    z_k = torch.FloatTensor(y.size(0), latent_dim).normal_(0, 1).to(device)

    # ODE function
    vpode = VPODE(ccf, y, save_path=save_path, plot=plot, every_n_plot=every_n_plot)
    states = (z_k,)
    integration_times = torch.linspace(vpode.T, 0., 2).type(torch.float32).to(device)

    # ODE solver
    odeint = odeint_adjoint if use_adjoint else odeint_normal
    state_t = odeint(
        vpode,
        states,
        integration_times,
        atol=atol,
        rtol=rtol,
        method=method)

    ccf.train()
    z_t0 = state_t[0][-1]
    print(f'#ODE steps for {y[0].item()}: {vpode.n_evals}')

    return z_t0.detach()


@register_sample_q
def sample_q_vpsde(ccf, y, device=torch.device('cuda'), save_path=None, plot=None, every_n_plot=5,
                   beta_min=0.1, beta_max=20, T=1, eps=1e-3, **kwargs):
    """sampling in the z space"""
    ccf.eval()

    latent_dim = kwargs['latent_dim']
    N = kwargs['N']
    correct_nsteps = kwargs['correct_nsteps']
    target_snr = kwargs['target_snr']

    # generate initial samples
    z_init = torch.FloatTensor(y.size(0), latent_dim).normal_(0, 1).to(device)
    z_k = torch.autograd.Variable(z_init, requires_grad=True)

    discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
    alphas = 1. - discrete_betas
    timesteps = torch.linspace(T, eps, N, device=device)

    # vpsde
    for k in range(N):

        if save_path is not None and k % every_n_plot == 0:
            g_z_sampled = ccf.g(z_k.detach())
            x_sampled = ccf.generate_images(g_z_sampled)
            plot('{}/samples_class{}_nsteps{}.png'.format(save_path, y[0].item(), k), x_sampled)

        energy_neg = ccf(z_k, y=y)

        # predictor
        t_k = timesteps[k]
        timestep = (t_k * (N - 1) / T).long()
        beta_t = discrete_betas[timestep]
        alpha_t = alphas[timestep]

        score_t = torch.autograd.grad(energy_neg.sum(), [z_k])[0]

        z_k = (2 - torch.sqrt(alpha_t)) * z_k + beta_t * score_t
        noise = torch.FloatTensor(y.size(0), latent_dim).normal_(0, 1).to(device)
        z_k = z_k + torch.sqrt(beta_t) * noise

        # corrector
        for j in range(correct_nsteps):
            noise = torch.FloatTensor(y.size(0), latent_dim).normal_(0, 1).to(device)

            grad_norm = torch.norm(score_t.reshape(score_t.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha_t

            assert step_size.ndim == 0, step_size.ndim

            z_k_mean = z_k + step_size * score_t
            z_k = z_k_mean + torch.sqrt(step_size * 2) * noise

    ccf.train()
    final_samples = z_k.detach()

    return final_samples
