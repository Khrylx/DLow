import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from models.mlp import MLP
from models.rnn import RNN
from utils.torch import *


class VAE(nn.Module):
    def __init__(self, nx, ny, nz, horizon, specs):
        super(VAE, self).__init__()
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.horizon = horizon
        self.rnn_type = rnn_type = specs.get('rnn_type', 'lstm')
        self.x_birnn = x_birnn = specs.get('x_birnn', True)
        self.e_birnn = e_birnn = specs.get('e_birnn', True)
        self.use_drnn_mlp = specs.get('use_drnn_mlp', False)
        self.nh_rnn = nh_rnn = specs.get('nh_rnn', 128)
        self.nh_mlp = nh_mlp = specs.get('nh_mlp', [300, 200])
        # encode
        self.x_rnn = RNN(nx, nh_rnn, bi_dir=x_birnn, cell_type=rnn_type)
        self.e_rnn = RNN(ny, nh_rnn, bi_dir=e_birnn, cell_type=rnn_type)
        self.e_mlp = MLP(2*nh_rnn, nh_mlp)
        self.e_mu = nn.Linear(self.e_mlp.out_dim, nz)
        self.e_logvar = nn.Linear(self.e_mlp.out_dim, nz)
        # decode
        if self.use_drnn_mlp:
            self.drnn_mlp = MLP(nh_rnn, nh_mlp + [nh_rnn], activation='tanh')
        self.d_rnn = RNN(ny + nz + nh_rnn, nh_rnn, cell_type=rnn_type)
        self.d_mlp = MLP(nh_rnn, nh_mlp)
        self.d_out = nn.Linear(self.d_mlp.out_dim, ny)
        self.d_rnn.set_mode('step')

    def encode_x(self, x):
        if self.x_birnn:
            h_x = self.x_rnn(x).mean(dim=0)
        else:
            h_x = self.x_rnn(x)[-1]
        return h_x

    def encode_y(self, y):
        if self.e_birnn:
            h_y = self.e_rnn(y).mean(dim=0)
        else:
            h_y = self.e_rnn(y)[-1]
        return h_y

    def encode(self, x, y):
        h_x = self.encode_x(x)
        h_y = self.encode_y(y)
        h = torch.cat((h_x, h_y), dim=1)
        h = self.e_mlp(h)
        return self.e_mu(h), self.e_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x, z):
        h_x = self.encode_x(x)
        if self.use_drnn_mlp:
            h_d = self.drnn_mlp(h_x)
            self.d_rnn.initialize(batch_size=z.shape[0], hx=h_d)
        else:
            self.d_rnn.initialize(batch_size=z.shape[0])
        y = []
        for i in range(self.horizon):
            y_p = x[-1] if i == 0 else y_i
            rnn_in = torch.cat([h_x, z, y_p], dim=1)
            h = self.d_rnn(rnn_in)
            h = self.d_mlp(h)
            y_i = self.d_out(h)
            y.append(y_i)
        y = torch.stack(y)
        return y

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar) if self.training else mu
        return self.decode(x, z), mu, logvar

    def sample_prior(self, x):
        z = torch.randn((x.shape[1], self.nz), device=x.device)
        return self.decode(x, z)


class NFDiag(nn.Module):
    def __init__(self, nx, ny, nk, specs):
        super(NFDiag, self).__init__()
        self.nx = nx
        self.ny = ny
        self.nk = nk
        self.nh = nh = specs.get('nh_mlp', [300, 200])
        self.nh_rnn = nh_rnn = specs.get('nh_rnn', 128)
        self.rnn_type = rnn_type = specs.get('rnn_type', 'gru')
        self.x_birnn = x_birnn = specs.get('x_birnn', False)
        self.fix_first = fix_first = specs.get('fix_first', False)
        self.nac = nac = nk - 1 if fix_first else nk
        self.x_rnn = RNN(nx, nh_rnn, bi_dir=x_birnn, cell_type=rnn_type)
        self.mlp = MLP(nh_rnn, nh)
        self.head_A = nn.Linear(nh[-1], ny * nac)
        self.head_b = nn.Linear(nh[-1], ny * nac)

    def encode_x(self, x):
        if self.x_birnn:
            h_x = self.x_rnn(x).mean(dim=0)
        else:
            h_x = self.x_rnn(x)[-1]
        return h_x

    def encode(self, x, y):
        if self.fix_first:
            z = y
        else:
            h_x = self.encode_x(x)
            h = self.mlp(h_x)
            a = self.head_A(h).view(-1, self.nk, self.ny)[:, 0, :]
            b = self.head_b(h).view(-1, self.nk, self.ny)[:, 0, :]
            z = (y - b) / a
        return z

    def forward(self, x, z=None):
        h_x = self.encode_x(x)
        if z is None:
            z = torch.randn((h_x.shape[0], self.ny), device=x.device)
        z = z.repeat_interleave(self.nk, dim=0)
        h = self.mlp(h_x)
        if self.fix_first:
            a = self.head_A(h).view(-1, self.nac, self.ny)
            b = self.head_b(h).view(-1, self.nac, self.ny)
            a = torch.cat((ones(h_x.shape[0], 1, self.ny, device=x.device), a), dim=1).view(-1, self.ny)
            b = torch.cat((zeros(h_x.shape[0], 1, self.ny, device=x.device), b), dim=1).view(-1, self.ny)
        else:
            a = self.head_A(h).view(-1, self.ny)
            b = self.head_b(h).view(-1, self.ny)
        y = a * z + b
        return y, a, b

    def sample(self, x, z=None):
        return self.forward(x, z)[0]

    def get_kl(self, a, b):
        var = a ** 2
        KLD = -0.5 * torch.sum(1 + var.log() - b.pow(2) - var)
        return KLD


class NFFull(nn.Module):
    def __init__(self, nx, ny, nk, specs):
        super(NFFull, self).__init__()
        self.nx = nx
        self.ny = ny
        self.nk = nk
        self.nh = nh = specs.get('nh_mlp', [300, 200])
        self.nh_rnn = nh_rnn = specs.get('nh_rnn', 128)
        self.rnn_type = rnn_type = specs.get('rnn_type', 'gru')
        self.x_birnn = x_birnn = specs.get('x_birnn', False)
        self.fix_first = fix_first = specs.get('fix_first', False)
        self.nac = nac = nk - 1 if fix_first else nk
        self.x_rnn = RNN(nx, nh_rnn, bi_dir=x_birnn, cell_type=rnn_type)
        self.mlp = MLP(nh_rnn, nh)
        self.head_A = nn.Linear(nh[-1], (ny * ny) * nac)
        self.head_b = nn.Linear(nh[-1], ny * nac)

    def encode_x(self, x):
        if self.x_birnn:
            h_x = self.x_rnn(x).mean(dim=0)
        else:
            h_x = self.x_rnn(x)[-1]
        return h_x

    def encode(self, x, y):
        z = y
        return z

    def forward(self, x, z=None):
        h_x = self.encode_x(x)
        if z is None:
            z = torch.randn((h_x.shape[0], self.ny, 1), device=x.device)
        else:
            z = z.unsqueeze(2)
        z = z.repeat_interleave(self.nk, dim=0)
        h = self.mlp(h_x)
        if self.fix_first:
            A = self.head_A(h).view(-1, self.nac, self.ny, self.ny)
            b = self.head_b(h).view(-1, self.nac, self.ny)
            cA = torch.eye(self.ny, device=x.device).repeat((h_x.shape[0], 1, 1, 1))
            A = torch.cat((cA, A), dim=1).view(-1, self.ny, self.ny)
            b = torch.cat((zeros(h_x.shape[0], 1, self.ny, device=x.device), b), dim=1).view(-1, self.ny)
        else:
            A = self.head_A(h).view(-1, self.ny, self.ny)
            b = self.head_b(h).view(-1, self.ny)
        y = A.bmm(z).squeeze(-1) + b
        return y, A, b

    def sample(self, x, z=None):
        return self.forward(x, z)[0]

    def get_kl(self, A, b):
        var = A.bmm(A.transpose(1, 2))
        KLD = -0.5 * (A.shape[-1] + torch.log(torch.det(var)) - b.pow(2).sum(dim=1) - (A * A).sum(dim=-1).sum(dim=-1))
        return KLD.sum()


def get_vae_model(cfg, traj_dim):
    specs = cfg.vae_specs
    model_name = specs.get('model_name', 'VAEv1')
    if model_name == 'VAEv1':
        return VAE(traj_dim, traj_dim, cfg.nz, cfg.t_pred, specs)


def get_dlow_model(cfg, traj_dim):
    specs = cfg.dlow_specs
    model_name = specs.get('model_name', 'NFDiag')
    if model_name == 'NFDiag':
        return NFDiag(traj_dim, cfg.nz, cfg.nk, specs)
    elif model_name == 'NFFull':
        return NFFull(traj_dim, cfg.nz, cfg.nk, specs)

