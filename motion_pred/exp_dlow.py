import os
import sys
import math
import pickle
import argparse
import time
from torch import optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())
from utils import *
from motion_pred.utils.config import Config
from motion_pred.utils.dataset_h36m import DatasetH36M
from motion_pred.utils.dataset_humaneva import DatasetHumanEva
from models.motion_pred import *


def joint_loss(Y_g):
    loss = 0.0
    Y_g = Y_g.permute(1, 0, 2).contiguous()
    Y_g = Y_g.view(Y_g.shape[0] // nk, nk, -1)
    for Y in Y_g:
        dist = F.pdist(Y, 2) ** 2
        loss += (-dist / cfg.d_scale).exp().mean()
    loss /= Y_g.shape[0]
    return loss


def recon_loss(Y_g, Y):
    Y_g = Y_g.view(Y_g.shape[0], -1, nk, Y_g.shape[2])
    diff = Y_g - Y.unsqueeze(2)
    dist = diff.pow(2).sum(dim=-1).sum(dim=0)
    loss_recon = dist.min(dim=1)[0].mean()
    return loss_recon


def loss_function(Y_g, Y, a, b):
    # KL divergence loss
    KLD = dlow.get_kl(a, b) / a.shape[0]
    # joint loss
    JL = joint_loss(Y_g) if cfg.lambda_j > 0 else 0.0
    RECON = recon_loss(Y_g, Y) if cfg.lambda_recon > 0 else 0.0
    loss_r = KLD * cfg.lambda_kl + JL * cfg.lambda_j + RECON * cfg.lambda_recon
    return loss_r, np.array([loss_r.item(), KLD.item(), 0 if JL == 0 else JL.item(), 0 if RECON == 0 else RECON.item()])


def train(epoch):
    t_s = time.time()
    train_losses = 0
    total_num_sample = 0
    loss_names = ['TOTAL', 'KLD', 'JL', 'RECON']
    generator = dataset.sampling_generator(num_samples=cfg.num_dlow_data_sample, batch_size=cfg.dlow_batch_size)
    for traj_np in generator:
        traj_np = traj_np[..., 1:, :].reshape(traj_np.shape[0], traj_np.shape[1], -1)
        traj = tensor(traj_np, device=device, dtype=dtype).permute(1, 0, 2).contiguous()
        X = traj[:t_his]
        Y = traj[t_his:]
        Z, a, b = dlow(X)
        if cfg.lambda_j > 0 or cfg.lambda_recon > 0:
            X_g = X.repeat_interleave(nk, dim=1)
            Y_g = vae.decode(X_g, Z)
        else:
            Y_g = None
        loss, losses = loss_function(Y_g, Y, a, b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses += losses
        total_num_sample += 1

    scheduler.step()
    dt = time.time() - t_s
    train_losses /= total_num_sample
    lr = optimizer.param_groups[0]['lr']
    losses_str = ' '.join(['{}: {:.4f}'.format(x, y) for x, y in zip(loss_names, train_losses)])
    logger.info('====> Epoch: {} Time: {:.2f} {} lr: {:.5f}'.format(epoch, dt, losses_str, lr))
    for name, loss in zip(loss_names, train_losses):
        tb_logger.add_scalar('dlow' + name, loss, epoch)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--mode', default='train')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_index', type=int, default=0)
    args = parser.parse_args()

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    cfg = Config(args.cfg, test=args.test)
    tb_logger = SummaryWriter(cfg.tb_dir) if args.mode == 'train' else None
    logger = create_logger(os.path.join(cfg.log_dir, 'log_dlow_recon.txt'))

    """parameter"""
    mode = args.mode
    nz = cfg.nz
    nk = cfg.nk
    t_his = cfg.t_his
    t_pred = cfg.t_pred

    """data"""
    dataset_cls = DatasetH36M if cfg.dataset == 'h36m' else DatasetHumanEva
    dataset = dataset_cls('train', t_his, t_pred, actions='all', use_vel=cfg.use_vel)
    if cfg.normalize_data:
        dataset.normalize_data()

    """model"""
    vae = get_vae_model(cfg, dataset.traj_dim)
    if not args.test:
        cp_path = cfg.vae_model_path % cfg.num_vae_epoch
        print('loading vae model from checkpoint: %s' % cp_path)
        vae_cp = pickle.load(open(cp_path, "rb"))
        vae.load_state_dict(vae_cp['model_dict'])

    dlow = get_dlow_model(cfg, dataset.traj_dim)
    optimizer = optim.Adam(dlow.parameters(), lr=cfg.dlow_lr)
    scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=cfg.num_dlow_epoch_fix, nepoch=cfg.num_dlow_epoch)

    if args.iter > 0:
        cp_path = cfg.dlow_model_path % args.iter
        print('loading dlow model from checkpoint: %s' % cp_path)
        dlow_cp = pickle.load(open(cp_path, "rb"))
        dlow.load_state_dict(dlow_cp['model_dict'])

    if mode == 'train':
        vae.to(device)
        vae.eval()
        dlow.to(device)
        dlow.train()
        for i in range(args.iter, cfg.num_dlow_epoch):
            train(i)
            if cfg.save_model_interval > 0 and (i + 1) % cfg.save_model_interval == 0:
                with to_cpu(dlow):
                    cp_path = cfg.dlow_model_path % (i + 1)
                    model_cp = {'model_dict': dlow.state_dict(), 'meta': {'std': dataset.std, 'mean': dataset.mean}}
                    pickle.dump(model_cp, open(cp_path, 'wb'))



