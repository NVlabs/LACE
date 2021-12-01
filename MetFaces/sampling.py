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

import dnnlib
import legacy

import utils
from metrics.id_loss import IDLoss
from metfaces_data import get_att_name_list, get_n_classes_list
from latent_model import DenseEmbedder


NETWORK_PKL = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl'


class F(nn.Module):
    def __init__(self, att_names='gender', latent_dim=512, truncation=1, subset_selection=False):
        super(F, self).__init__()

        self.ws_prev = None
        self.subset_selection = subset_selection

        print('Loading networks from "%s"...' % NETWORK_PKL)
        with dnnlib.util.open_url(NETWORK_PKL) as f:
            G = legacy.load_network_pkl(f)['G_ema']  # type: ignore
        self.G = G

        self.truncation = truncation
        print(f'truncation: {truncation}')

        att_names = get_att_name_list(att_names, for_model=True)
        self.att_names = att_names
        self.n_classes_list = get_n_classes_list(att_names)
        self.n_att = len(self.n_classes_list)
        print(f'att_names: {att_names}, n_classes_list: {self.n_classes_list}, num_attributes: {self.n_att}')

        self.f = []
        for i, att_name in enumerate(att_names):
            f_i = DenseEmbedder(input_dim=latent_dim, up_dim=128, norm=None,
                                num_classes_list=get_n_classes_list(att_name))
            self.f.append(f_i)

        def mapping_z_to_w(z):
            assert z.shape[-1] == G.z_dim
            label = torch.zeros([z.shape[0], self.G.c_dim])
            ws = G.mapping(z, label, truncation_psi=truncation)
            return ws[:, 0, :]

        self.g = mapping_z_to_w

    def classify(self, z, seq_indices=[]):
        f_seq = self.f
        if len(seq_indices) > 0:  # choose a subset of attributes for classification
            f_seq = [self.f[i] for i in seq_indices]
        logits_list = []
        for f_i in f_seq:
            logits_list.extend(f_i(self.g(z)))

        return logits_list

    def calc_logits_dist(self, z_0, z, idxes=[]):
        if len(idxes) == 0:
            return None

        f_seq = [self.f[i] for i in idxes]
        logits_list, logits_0_list = [], []
        for f_i in f_seq:
            logits_list.extend(f_i(self.g(z)))
            logits_0_list.extend(f_i(self.g(z_0.detach())))

        logits_dist = 0
        for logits, logits_0 in zip(logits_list, logits_0_list):
            logits_dist += torch.linalg.norm(logits - logits_0, dim=1) ** 2
        assert logits_dist.ndim == 1, logits_dist.ndim
        return logits_dist

    def classify_x(self, x):
        logits_list = []
        for f_i in self.f:
            logits_list.extend(f_i(x))
        return logits_list

    def apply_subset_selection(self, ws, seq_indices=[], update_ws_prev=False):
        if self.ws_prev is not None:
            att_names_cur = [self.att_names[i] for i in seq_indices]
            subset_nosel = utils.subset_from_att_names(att_names_cur)
            print(f'subset_nosel: {subset_nosel} with seq_indices: {seq_indices}')

            ws[:, subset_nosel, :] = self.ws_prev[:, subset_nosel, :]

        if update_ws_prev:
            self.ws_prev = ws.detach().clone()

        return ws

    def generate_images(self, g_z_sampled, seq_indices=[], update_ws_prev=False, is_detached=True):
        """Generate images from g_z_sampled"""
        ws = g_z_sampled
        if ws.ndim == 2:
            ws = ws.unsqueeze(1).repeat([1, 18, 1])  # bs x n_mappings x 512

        if self.subset_selection:
            ws = self.apply_subset_selection(ws, seq_indices, update_ws_prev=update_ws_prev)

        assert ws.shape[1:] == (self.G.num_ws, self.G.w_dim)
        img = self.G.synthesis(ws, noise_mode='const')
        return img.detach() if is_detached else img


class CCF(F):
    def __init__(self, att_names='gender', expr='', latent_dim=512, truncation=1, subset_selection=False):
        super(CCF, self).__init__(att_names, latent_dim=latent_dim, truncation=truncation,
                                  subset_selection=subset_selection)
        self.expr = expr

    def _get_single_cond_energy(self, logits, y, dis_temp=1.):
        assert y.ndim == 1 and logits.ndim == 2, (y.ndim, logits.ndim)
        n_classes = logits.size(1)
        if n_classes > 1:  # discrete attribute
            y = y.long()
            single_cond_energy = torch.gather(logits / dis_temp, 1, y[:, None]).squeeze() - logits.logsumexp(1)
        else:  # continuous attribute
            assert n_classes == 1, n_classes
            y = y.float()
            sigma = 0.1  # this value works well
            single_cond_energy = -torch.linalg.norm(logits - y[:, None], dim=1) ** 2 * 0.5 / (sigma ** 2)
        assert single_cond_energy.ndim == 1, single_cond_energy.ndim
        return single_cond_energy

    def get_cond_energy(self, z, ys, seq_indices=[], reweight=True, dis_temp=1.):
        logits_list = self.classify(z, seq_indices=seq_indices)

        weight_list = [1. for _ in range(len(logits_list))]
        if len(seq_indices) > 0 and reweight:  # seq edit only
            weight_list = [0.1 for _ in range(len(logits_list) - 1)]
            weight_list += [10.] if self.n_classes_list[seq_indices[-1]] == 1 else [1.]

        energy_outs = []
        for j, logits in enumerate(logits_list):
            energy_neg_cond = self._get_single_cond_energy(logits, ys[:, j], dis_temp)  # ys: [bs, n_attributes]
            energy_outs.append(energy_neg_cond * weight_list[j])

        if self.expr != '':  # w/ logical operators
            energy_output = utils.logical_comb(self.expr, energy_outs)
        else:  # w/o logical operators
            energy_output = torch.stack(energy_outs).sum(dim=0)

        return energy_output

    def forward(self, z, ys, seq_indices=[], reweight=True, dis_temp=1.):
        energy_output = self.get_cond_energy(z, ys, seq_indices=seq_indices, reweight=reweight, dis_temp=dis_temp) \
                        - torch.linalg.norm(z, dim=1) ** 2 * 0.5
        return energy_output


# ----------------------------------------------------------------------------

_sample_q_dict = dict()  # name => fn


def register_sample_q(fn):
    assert callable(fn)
    _sample_q_dict[fn.__name__] = fn
    return fn


@register_sample_q
def sample_q_sgld(ccf, ys, device=torch.device('cuda'), save_path=None, plot=None, every_n_plot=5,
                  every_n_print=5, reg_z=0, z_init=None, z_anchor=None, seq_indices=[], reg_id=0,
                  reg_logits=0, reweight=True, dis_temp=1., **kwargs):
    """sampling in the z space"""
    ccf.eval()

    latent_dim = kwargs['latent_dim']
    sgld_lr = kwargs['sgld_lr']
    sgld_std = kwargs['sgld_std']
    n_steps = kwargs['n_steps']

    ys_seq = ys[0].cpu().numpy()

    # generate initial samples
    if z_init is None:
        z_init = torch.FloatTensor(ys.size(0), latent_dim).normal_(0, 1).to(device)
    z_k = torch.autograd.Variable(z_init, requires_grad=True)

    if z_anchor is None:
        z_anchor = z_init.detach().clone()

    # sgld
    for k in range(n_steps):

        if save_path is not None and k % every_n_plot == 0:
            g_z_sampled = ccf.g(z_k.detach())
            x_sampled = ccf.generate_images(g_z_sampled, seq_indices=seq_indices)
            plot(f'{save_path}/samples_indices{seq_indices}_cls{ys_seq}_nsteps{k}.png', x_sampled)

        energy_neg = ccf(z_k, ys=ys, seq_indices=seq_indices, reweight=reweight, dis_temp=dis_temp)

        # ---------------- Add regularizations for sequential editing -------------
        z_diff_norm = None
        if reg_z > 0:
            assert z_anchor is not None
            z_diff_norm = torch.linalg.norm(ccf.g(z_k) - ccf.g(z_anchor), dim=1) ** 2 + \
                          torch.linalg.norm(z_k - z_anchor, dim=1) ** 2

            assert z_diff_norm.ndim == energy_neg.ndim == 1, (z_diff_norm.ndim, energy_neg.ndim)
            energy_neg -= reg_z * z_diff_norm

        logits_dist = None
        if reg_logits > 0:
            idxes_remain = [i for i in range(ccf.n_att) if i not in seq_indices]
            logits_dist = ccf.calc_logits_dist(z_0=z_anchor, z=z_k, idxes=idxes_remain)
            if logits_dist is not None:
                energy_neg -= reg_logits * logits_dist
        # ---------------- [END] Add regularizations for sequential editing -------------

        f_prime = torch.autograd.grad(energy_neg.sum(), [z_k])[0]
        z_k.data += sgld_lr * f_prime + sgld_std * torch.randn_like(z_k)

        if save_path is not None and k % every_n_print == 0:
            print(f'f_prime (mean, var): {f_prime.mean().item(), f_prime.var().item()}, '
                  f'z_diff_norm mean: {0 if z_diff_norm is None else z_diff_norm.mean().item()}, '
                  f'logits_dist mean: {0 if logits_dist is None else logits_dist.mean().item()}, '
                  f'energy_neg mean: {energy_neg.mean().item()} with cls{ys_seq}')

    ccf.train()
    final_samples = z_k.detach()

    return final_samples


class VPODE(nn.Module):
    def __init__(self, ccf, ys, beta_min=0.1, beta_max=20, T=1.0, save_path=None, plot=None,
                 device=torch.device('cuda'), every_n_plot=5, every_n_print=5, reg_z=0, seq_indices=[],
                 reg_id=0, reg_logits=0, reweight=True, dis_temp=1.):
        super().__init__()
        self.ccf = ccf
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.T = T
        self.ys = ys
        self.save_path = save_path
        self.n_evals = 0
        self.every_n_plot = every_n_plot
        self.every_n_print = every_n_print
        self.plot = plot

        self.z_anchor = None
        self.reg_z = reg_z
        self.seq_indices = seq_indices
        self.ys_seq = ys[0].cpu().numpy()
        print(f'ys_seq: {self.ys_seq}')

        self.reweight = reweight
        self.dis_temp = dis_temp

        if reg_id > 0:
            self.id_loss = IDLoss().to(device).eval()

        self.reg_id = reg_id
        self.reg_logits = reg_logits

        print(f'(reg_z, reg_id, reg_logits) in vpode: {self.reg_z, self.reg_id, self.reg_logits}')

    def set_anchor_z(self, z):
        self.z_anchor = None if z is None else z.detach().clone()

    def forward(self, t_k, states):
        z = states[0]

        if self.save_path is not None and self.n_evals % self.every_n_plot == 0:
            g_z_sampled = self.ccf.g(z.detach())
            x_sampled = self.ccf.generate_images(g_z_sampled, seq_indices=self.seq_indices)
            self.plot(f'{self.save_path}/samples_indices{self.seq_indices}_cls{self.ys_seq}_'
                      f'nsteps{self.n_evals:03d}_tk{t_k}.png', x_sampled)

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)

            beta_t = self.beta_0 + t_k * (self.beta_1 - self.beta_0)
            cond_energy_neg = self.ccf.get_cond_energy(z, self.ys, self.seq_indices, self.reweight, self.dis_temp)

            # ---------------- Add regularizations for sequential editing -------------
            z_diff_norm = None
            if self.reg_z > 0:
                assert self.z_anchor is not None

                z_diff_norm = torch.linalg.norm(self.ccf.g(z) - self.ccf.g(self.z_anchor), dim=1) ** 2 + \
                              torch.linalg.norm(z - self.z_anchor, dim=1) ** 2

                assert z_diff_norm.ndim == cond_energy_neg.ndim == 1, (z_diff_norm.ndim, cond_energy_neg.ndim)
                cond_energy_neg -= self.reg_z * z_diff_norm

            id_loss = None
            if self.reg_id > 0:
                x = self.ccf.generate_images(self.ccf.g(z), seq_indices=self.seq_indices, is_detached=False)
                x_anchor = self.ccf.generate_images(self.ccf.g(self.z_anchor), seq_indices=self.seq_indices)
                id_loss = self.id_loss(x_anchor=x_anchor, x=x)
                cond_energy_neg -= self.reg_id * id_loss

            logits_dist = None
            if self.reg_logits > 0:
                idxes_remain = [i for i in range(self.ccf.n_att) if i not in self.seq_indices]
                logits_dist = self.ccf.calc_logits_dist(z_0=self.z_anchor, z=z, idxes=idxes_remain)
                if logits_dist is not None:
                    cond_energy_neg -= self.reg_logits * logits_dist
            # ---------------- [End] Add regularizations for sequential editing -------------

            cond_f_prime = torch.autograd.grad(cond_energy_neg.sum(), [z])[0]
            dz_dt = -0.5 * beta_t * cond_f_prime

            if self.save_path is not None and self.n_evals % self.every_n_print == 0:
                print(f'cond_f_prime (mean, var): {cond_f_prime.mean().item(), cond_f_prime.var().item()}, '
                      f'z_diff_norm mean: {0 if z_diff_norm is None else z_diff_norm.mean().item()}, '
                      f'id_loss mean: {0 if id_loss is None else id_loss.mean().item()}, '
                      f'logits_dist mean: {0 if logits_dist is None else logits_dist.mean().item()}, '
                      f'cond_energy_neg mean: {cond_energy_neg.mean().item()} with cls{self.ys_seq}')

        self.n_evals += 1

        return dz_dt,


@register_sample_q
def sample_q_ode(ccf, ys, device=torch.device('cuda'), save_path=None, plot=None, every_n_plot=5, every_n_print=5,
                 reg_z=0, z_init=None, z_anchor=None, seq_indices=[], reg_id=0, reg_logits=0, reweight=True,
                 dis_temp=1., **kwargs):
    """sampling in the z space"""
    ccf.eval()

    latent_dim = kwargs['latent_dim']
    atol = kwargs['atol']
    rtol = kwargs['rtol']
    method = kwargs['method']
    use_adjoint = kwargs['use_adjoint']

    # generate initial samples
    if z_init is None:
        z_init = torch.FloatTensor(ys.size(0), latent_dim).normal_(0, 1).to(device)
    if z_anchor is None:
        z_anchor = z_init

    # ODE function
    vpode = VPODE(ccf, ys, save_path=save_path, plot=plot, device=device, every_n_plot=every_n_plot,
                  every_n_print=every_n_print, reg_z=reg_z, seq_indices=seq_indices, reg_id=reg_id,
                  reg_logits=reg_logits, reweight=reweight, dis_temp=dis_temp).to(device)
    vpode.set_anchor_z(z_anchor)
    states = (z_init,)
    integration_times = torch.linspace(vpode.T, 1e-3, 2).type(torch.float32).to(device)

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
    print(f'#ODE steps for {vpode.ys_seq}: {vpode.n_evals}')

    return z_t0.detach()


@register_sample_q
def sample_q_vpsde(ccf, ys, device=torch.device('cuda'), save_path=None, plot=None, every_n_plot=5,
                   every_n_print=5, reg_z=0, z_init=None, z_anchor=None, seq_indices=[], reg_id=0,
                   reg_logits=0, reweight=True, dis_temp=1., beta_min=0.1, beta_max=20, T=1, eps=1e-3, **kwargs):
    """sampling in the z space"""
    ccf.eval()

    latent_dim = kwargs['latent_dim']
    N = kwargs['N']
    correct_nsteps = kwargs['correct_nsteps']
    target_snr = kwargs['target_snr']

    ys_seq = ys[0].cpu().numpy()

    # generate initial samples
    if z_init is None:
        z_init = torch.FloatTensor(ys.size(0), latent_dim).normal_(0, 1).to(device)
    z_k = torch.autograd.Variable(z_init, requires_grad=True)

    if z_anchor is None:
        z_anchor = z_init.detach().clone()

    discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
    alphas = 1. - discrete_betas
    timesteps = torch.linspace(T, eps, N, device=device)

    # vpsde
    for k in range(N):

        if save_path is not None and k % every_n_plot == 0:
            g_z_sampled = ccf.g(z_k.detach())
            x_sampled = ccf.generate_images(g_z_sampled, seq_indices=seq_indices)
            plot(f'{save_path}/samples_indices{seq_indices}_cls{ys_seq}_nsteps{k}.png', x_sampled)

        energy_neg = ccf(z_k, ys=ys, seq_indices=seq_indices, reweight=reweight, dis_temp=dis_temp)

        # ---------------- Add regularizations for sequential editing -------------
        z_diff_norm = None
        if reg_z > 0:
            assert z_anchor is not None

            z_diff_norm = torch.linalg.norm(ccf.g(z_k) - ccf.g(z_anchor), dim=1) ** 2 + \
                          torch.linalg.norm(z_k - z_anchor, dim=1) ** 2

            assert z_diff_norm.ndim == energy_neg.ndim == 1, (z_diff_norm.ndim, energy_neg.ndim)
            energy_neg -= reg_z * z_diff_norm

        logits_dist = None
        if reg_logits > 0:
            idxes_remain = [i for i in range(ccf.n_att) if i not in seq_indices]
            logits_dist = ccf.calc_logits_dist(z_0=z_anchor, z=z_k, idxes=idxes_remain)
            if logits_dist is not None:
                energy_neg -= reg_logits * logits_dist
        # ---------------- [END] Add regularizations for sequential editing -------------

        # predictor
        t_k = timesteps[k]
        timestep = (t_k * (N - 1) / T).long()
        beta_t = discrete_betas[timestep]
        alpha_t = alphas[timestep]

        score_t = torch.autograd.grad(energy_neg.sum(), [z_k])[0]

        z_k = (2 - torch.sqrt(alpha_t)) * z_k + beta_t * score_t
        noise = torch.FloatTensor(ys.size(0), latent_dim).normal_(0, 1).to(device)
        z_k = z_k + torch.sqrt(beta_t) * noise

        # corrector
        for j in range(correct_nsteps):
            noise = torch.FloatTensor(ys.size(0), latent_dim).normal_(0, 1).to(device)

            grad_norm = torch.norm(score_t.reshape(score_t.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha_t

            assert step_size.ndim == 0, step_size.ndim

            z_k_mean = z_k + step_size * score_t
            z_k = z_k_mean + torch.sqrt(step_size * 2) * noise

        if save_path is not None and k % every_n_print == 0:
            print(f'step size: {step_size.item()}')

            print(f'score_t (mean, var): {score_t.mean().item(), score_t.var().item()}, '
                  f'z_diff_norm mean: {0 if z_diff_norm is None else z_diff_norm.mean().item()}, '
                  f'logits_dist mean: {0 if logits_dist is None else logits_dist.mean().item()}, '
                  f'energy_neg mean: {energy_neg.mean().item()} with cls{ys_seq}')

    ccf.train()
    final_samples = z_k.detach()

    return final_samples
