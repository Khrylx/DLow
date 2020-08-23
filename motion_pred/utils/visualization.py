# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def render_animation(skeleton, poses_generator, algos, t_hist, fix_0=True, azim=0.0, output=None, size=6, ncol=5, bitrate=3000):
    """
    TODO
    Render an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """

    all_poses = next(poses_generator)
    algo = algos[0] if len(algos) > 0 else next(iter(all_poses.keys()))
    t_total = next(iter(all_poses.values())).shape[0]
    poses = dict(filter(lambda x: x[0] in {'gt', 'context'} or algo == x[0].split('_')[0], all_poses.items()))
    plt.ioff()
    nrow = int(np.ceil(len(poses) / ncol))
    fig = plt.figure(figsize=(size*ncol, size*nrow))
    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 1.7
    for index, (title, data) in enumerate(poses.items()):
        ax = fig.add_subplot(nrow, ncol, index+1, projection='3d')
        ax.view_init(elev=15., azim=azim)
        ax.set_xlim3d([-radius/2, radius/2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius/2, radius/2])
        ax.set_aspect('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 5.0
        ax.set_title(title, y=1.2)
        ax.set_axis_off()
        ax.patch.set_alpha(0.0)
        ax_3d.append(ax)
        lines_3d.append([])
        trajectories.append(data[:, 0, [0, 1]])
    fig.tight_layout()
    fig.subplots_adjust(wspace=-0.4, hspace=0)
    poses = list(poses.values())

    anim = None
    initialized = False
    animating = True
    find = 0
    hist_lcol, hist_rcol = 'black', 'red'
    pred_lcol, pred_rcol = 'purple', 'green'

    parents = skeleton.parents()

    def update_video(i):
        nonlocal initialized
        if i < t_hist:
            lcol, rcol = hist_lcol, hist_rcol
        else:
            lcol, rcol = pred_lcol, pred_rcol

        for n, ax in enumerate(ax_3d):
            if fix_0 and n == 0 and i >= t_hist:
                continue
            trajectories[n] = poses[n][:, 0, [0, 1, 2]]
            ax.set_xlim3d([-radius/2 + trajectories[n][i, 0], radius/2 + trajectories[n][i, 0]])
            ax.set_ylim3d([-radius/2 + trajectories[n][i, 1], radius/2 + trajectories[n][i, 1]])
            ax.set_zlim3d([-radius/2 + trajectories[n][i, 2], radius/2 + trajectories[n][i, 2]])

        if not initialized:

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                col = rcol if j in skeleton.joints_right() else lcol
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                               [pos[j, 1], pos[j_parent, 1]],
                                               [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col))
            initialized = True
        else:

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                col = rcol if j in skeleton.joints_right() else lcol
                for n, ax in enumerate(ax_3d):
                    if fix_0 and n == 0 and i >= t_hist:
                        continue
                    pos = poses[n][i]
                    lines_3d[n][j-1][0].set_xdata([pos[j, 0], pos[j_parent, 0]])
                    lines_3d[n][j-1][0].set_ydata([pos[j, 1], pos[j_parent, 1]])
                    lines_3d[n][j-1][0].set_3d_properties([pos[j, 2], pos[j_parent, 2]], zdir='z')
                    lines_3d[n][j-1][0].set_color(col)

    def show_animation():
        nonlocal anim
        if anim is not None:
            anim.event_source.stop()
        anim = FuncAnimation(fig, update_video, frames=np.arange(0, poses[0].shape[0]), interval=0, repeat=True)
        plt.draw()

    def reload_poses():
        nonlocal poses
        poses = dict(filter(lambda x: x[0] in {'gt', 'context'} or algo == x[0].split('_')[0], all_poses.items()))
        for ax, title in zip(ax_3d, poses.keys()):
            ax.set_title(title, y=1.2)
        poses = list(poses.values())

    def save_figs():
        nonlocal algo, find
        old_algo = algo
        for algo in algos:
            reload_poses()
            update_video(t_total - 1)
            fig.savefig('out/%d_%s.png' % (find, algo), dpi=400, transparent=True)
        algo = old_algo
        find += 1

    def on_key(event):
        nonlocal algo, all_poses, animating, anim

        if event.key == 'd':
            all_poses = next(poses_generator)
            reload_poses()
            show_animation()
        elif event.key == 'c':
            save()
        elif event.key == ' ':
            if animating:
                anim.event_source.stop()
            else:
                anim.event_source.start()
            animating = not animating
        elif event.key == 'v':  # save images
            if anim is not None:
                anim.event_source.stop()
                anim = None
            save_figs()
        elif event.key.isdigit():
            algo = algos[int(event.key) - 1]
            reload_poses()
            show_animation()

    def save():
        nonlocal anim

        fps = 30
        anim = FuncAnimation(fig, update_video, frames=np.arange(0, poses[0].shape[0]), interval=1000 / fps, repeat=False)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        if output.endswith('.mp4'):
            Writer = writers['ffmpeg']
            writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
            anim.save(output, writer=writer)
        elif output.endswith('.gif'):
            anim.save(output, dpi=80, writer='imagemagick')
        else:
            raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
        print(f'video saved to {output}!')

    fig.canvas.mpl_connect('key_press_event', on_key)
    show_animation()
    plt.show()
