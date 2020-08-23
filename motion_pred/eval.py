import numpy as np
import argparse
import os
import sys
import pickle
import csv
from scipy.spatial.distance import pdist

sys.path.append(os.getcwd())
from utils import *
from motion_pred.utils.config import Config
from motion_pred.utils.dataset_h36m import DatasetH36M
from motion_pred.utils.dataset_humaneva import DatasetHumanEva
from motion_pred.utils.visualization import render_animation
from models.motion_pred import *
from scipy.spatial.distance import pdist, squareform


def denomarlize(*data):
    out = []
    for x in data:
        x = x * dataset.std + dataset.mean
        out.append(x)
    return out


def get_prediction(data, algo, sample_num, num_seeds=1, concat_hist=True):
    traj_np = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
    traj = tensor(traj_np, device=device, dtype=dtype).permute(1, 0, 2).contiguous()
    X = traj[:t_his]

    if algo == 'dlow':
        X = X.repeat((1, num_seeds, 1))
        Z_g = models[algo].sample(X)
        X = X.repeat_interleave(nk, dim=1)
        Y = models['vae'].decode(X, Z_g)
    elif algo == 'vae':
        X = X.repeat((1, sample_num * num_seeds, 1))
        Y = models[algo].sample_prior(X)

    if concat_hist:
        Y = torch.cat((X, Y), dim=0)
    Y = Y.permute(1, 0, 2).contiguous().cpu().numpy()
    if Y.shape[0] > 1:
        Y = Y.reshape(-1, sample_num, Y.shape[-2], Y.shape[-1])
    else:
        Y = Y[None, ...]
    return Y


def visualize():

    def post_process(pred, data):
        pred = pred.reshape(pred.shape[0], pred.shape[1], -1, 3)
        if cfg.normalize_data:
            pred = denomarlize(pred)
        pred = np.concatenate((np.tile(data[..., :1, :], (pred.shape[0], 1, 1, 1)), pred), axis=2)
        pred[..., :1, :] = 0
        return pred

    def pose_generator():

        while True:
            data = dataset.sample()

            # gt
            gt = data[0].copy()
            gt[:, :1, :] = 0
            poses = {'context': gt, 'gt': gt}
            # vae
            for algo in vis_algos:
                pred = get_prediction(data, algo, nk)[0]
                pred = post_process(pred, data)
                for i in range(pred.shape[0]):
                    poses[f'{algo}_{i}'] = pred[i]

            yield poses

    pose_gen = pose_generator()
    render_animation(dataset.skeleton, pose_gen, vis_algos, cfg.t_his, ncol=12, output='out/video.mp4')


def get_gt(data):
    gt = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
    return gt[:, t_his:, :]


"""metrics"""

def compute_diversity(pred, *args):
    if pred.shape[0] == 1:
        return 0.0
    dist = pdist(pred.reshape(pred.shape[0], -1))
    diversity = dist.mean().item()
    return diversity


def compute_ade(pred, gt, *args):
    diff = pred - gt
    dist = np.linalg.norm(diff, axis=2).mean(axis=1)
    return dist.min()


def compute_fde(pred, gt, *args):
    diff = pred - gt
    dist = np.linalg.norm(diff, axis=2)[:, -1]
    return dist.min()


def compute_mmade(pred, gt, gt_multi):
    gt_dist = []
    for gt_multi_i in gt_multi:
        dist = compute_ade(pred, gt_multi_i)
        gt_dist.append(dist)
    gt_dist = np.array(gt_dist).mean()
    return gt_dist


def compute_mmfde(pred, gt, gt_multi):
    gt_dist = []
    for gt_multi_i in gt_multi:
        dist = compute_fde(pred, gt_multi_i)
        gt_dist.append(dist)
    gt_dist = np.array(gt_dist).mean()
    return gt_dist


def compute_stats():
    stats_func = {'Diversity': compute_diversity, 'ADE': compute_ade,
                  'FDE': compute_fde, 'MMADE': compute_mmade, 'MMFDE': compute_mmfde}
    stats_names = list(stats_func.keys())
    stats_meter = {x: {y: AverageMeter() for y in algos} for x in stats_names}

    data_gen = dataset.iter_generator(step=cfg.t_his)
    num_samples = 0
    num_seeds = args.num_seeds
    for i, data in enumerate(data_gen):
        num_samples += 1
        gt = get_gt(data)
        gt_multi = traj_gt_arr[i]
        for algo in algos:
            pred = get_prediction(data, algo, sample_num=cfg.nk, num_seeds=num_seeds, concat_hist=False)
            for stats in stats_names:
                val = 0
                for pred_i in pred:
                    val += stats_func[stats](pred_i, gt, gt_multi) / num_seeds
                stats_meter[stats][algo].update(val)
        print('-' * 80)
        for stats in stats_names:
            str_stats = f'{num_samples:04d} {stats}: ' + ' '.join([f'{x}: {y.val:.4f}({y.avg:.4f})' for x, y in stats_meter[stats].items()])
            print(str_stats)

    logger.info('=' * 80)
    for stats in stats_names:
        str_stats = f'Total {stats}: ' + ' '.join([f'{x}: {y.avg:.4f}' for x, y in stats_meter[stats].items()])
        logger.info(str_stats)
    logger.info('=' * 80)

    with open('%s/stats_%s.csv' % (cfg.result_dir, args.num_seeds), 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['Metric'] + algos)
        writer.writeheader()
        for stats, meter in stats_meter.items():
            new_meter = {x: y.avg for x, y in meter.items()}
            new_meter['Metric'] = stats
            writer.writerow(new_meter)


def get_multimodal_gt():
    all_data = []
    data_gen = dataset.iter_generator(step=cfg.t_his)
    for data in data_gen:
        data = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
        all_data.append(data)
    all_data = np.concatenate(all_data, axis=0)
    all_start_pose = all_data[:, t_his - 1, :]
    pd = squareform(pdist(all_start_pose))
    traj_gt_arr = []
    for i in range(pd.shape[0]):
        ind = np.nonzero(pd[i] < args.multimodal_threshold)
        traj_gt_arr.append(all_data[ind][:, t_his:, :])
    return traj_gt_arr


if __name__ == '__main__':

    all_algos = ['dlow', 'vae']
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--mode', default='vis')
    parser.add_argument('--data', default='test')
    parser.add_argument('--action', default='all')
    parser.add_argument('--num_seeds', type=int, default=1)
    parser.add_argument('--multimodal_threshold', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_index', type=int, default=-1)
    for algo in all_algos:
        parser.add_argument('--iter_%s' % algo, type=int, default=None)
    args = parser.parse_args()

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if args.gpu_index >= 0 and torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    torch.set_grad_enabled(False)
    cfg = Config(args.cfg)
    logger = create_logger(os.path.join(cfg.log_dir, 'log_eval.txt'))

    algos = []
    for algo in all_algos:
        iter_algo = 'iter_%s' % algo
        num_algo = 'num_%s_epoch' % algo
        setattr(args, iter_algo, getattr(cfg, num_algo))
        algos.append(algo)
    vis_algos = algos.copy()

    if args.action != 'all':
        args.action = set(args.action.split(','))

    """parameter"""
    nz = cfg.nz
    nk = cfg.nk
    t_his = cfg.t_his
    t_pred = cfg.t_pred

    """data"""
    dataset_cls = DatasetH36M if cfg.dataset == 'h36m' else DatasetHumanEva
    dataset = dataset_cls(args.data, t_his, t_pred, actions=args.action, use_vel=cfg.use_vel)
    traj_gt_arr = get_multimodal_gt()

    """models"""
    model_generator = {
        'vae': get_vae_model,
        'dlow': get_dlow_model,
    }
    models = {}
    for algo in algos:
        models[algo] = model_generator[algo](cfg, dataset.traj_dim)
        model_path = getattr(cfg, f"{algo}_model_path") % getattr(args, f'iter_{algo}')
        print(f'loading {algo} model from checkpoint: {model_path}')
        model_cp = pickle.load(open(model_path, "rb"))
        models[algo].load_state_dict(model_cp['model_dict'])
        models[algo].to(device)
        models[algo].eval()

    if cfg.normalize_data:
        dataset.normalize_data(model_cp['meta']['mean'], model_cp['meta']['std'])

    if args.mode == 'vis':
        visualize()
    elif args.mode == 'stats':
        compute_stats()
