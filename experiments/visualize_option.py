#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Python imports.
import sys
import logging
import numpy as np
import tensorflow as tf
import random

from scipy.misc import imread

# Other imports.
# import srl_example_setup
from simple_rl.agents import RandomAgent, DDPGAgent, DQNAgent, LinearQAgent
from simple_rl.agents.func_approx.Features import Fourier, Monte, Subset
from simple_rl.tasks import GymMDP, PinballMDP
from simple_rl.tasks.pinball.PinballStateClass import PinballState
from simple_rl.agents.func_approx.ExperienceBuffer import ExperienceBuffer

from simple_rl.run_experiments import run_agents_on_mdp

from options.OptionWrapper import OptionWrapper
from options.OptionAgent import OptionAgent

from util import get_mdp_params, arguments, sample_option_trajectories

def plot_eigenfunction(op, args, xind=0, yind=1, filename='visualize_ef.pdf'):
    # Pinball

    mdp, state_dim, state_bound, num_actions, action_dim, action_bound = get_mdp_params(args)
    
    n_samples = 2000
    
    low_bound = state_bound[0]
    up_bound = state_bound[1]

    if args.task == 'AntMaze-v0' or args.task == 'PointMaze-v0':
        low_bound[xind] = 0.0
        low_bound[yind] = 0.0
        up_bound[xind] = 8.0 * 3.0
        up_bound[yind] = 8.0 * 3.0

    if args.tasktype == 'atari':
        low_bound[xind] = 0.0
        low_bound[yind] = 0.0
        up_bound[xind] = 160.0
        up_bound[yind] = 210.0

    xs = []
    ys = []
    fs = []

    # if np.isinf(low_bound).any() or np.isinf(up_bound).any():
    #     bfr = sample_option_trajectories(mdp, args, noptions=0)
    # 
    #     ss, _, _, _, _ = bfr.sample(n_samples)
    # 
    #     max_x = float('-inf')
    #     min_x = float('inf')
    #     max_y = float('-inf')
    #     min_y = float('inf')
    #     for i in range(n_samples):
    #         x = ss[i].data[xind]
    #         y = ss[i].data[yind]
    #         max_x = max(x, max_x)
    #         min_x = min(x, min_x)
    #         max_y = max(y, max_y)
    #         min_y = min(y, min_y)
    #     low_bound[xind] = min_x
    #     up_bound[xind] = max_x
    #     low_bound[yind] = min_y
    #     up_bound[yind] = max_y

    # TODO: Implement a script to plot the f-value of the states
    #       visited by the agent instead of sampling uniform randomly.

    if args.restoretraj:
        print('restoring buffer from ' + './vis/' + args.task + 'option' + str(args.noptions - 1) + '_' + str(args.ffuncnunit) + '_' + str(args.rseed) + '/' + 'traj')
        bfr = ExperienceBuffer()
        bfr.restore(args.basedir + '/vis/' + args.task + 'option' + str(args.noptions - 1) + '_' + str(args.ffuncnunit) + '_' + str(args.rseed) + '/' + 'traj')
        bfr_size = bfr.size()
        print('bfr_size=', bfr_size) # TODO: parameter?

        samples, _, _, _, _ = bfr.sample(n_samples)
        # samples = [bfr.buffer[i][0] for i in range(min(bfr.size(), n_samples))]

        if args.task == 'MontezumaRevenge-ram-v0':
            feature = Monte()
            xs = [feature.feature(s, 0)[0] for s in samples]
            ys = [feature.feature(s, 0)[1] for s in samples]
        elif args.ffunction == 'nns':
            feature = Subset(state_dim, [0, 1])
            xs = [feature.feature(s, 0)[0] for s in samples]
            ys = [feature.feature(s, 0)[1] for s in samples]
        else:
            xs = [s.data[xind] for s in samples]
            ys = [s.data[yind] for s in samples]


         
    else:        
        xs = [random.uniform(low_bound[xind], up_bound[xind]) for _ in range(n_samples)]
        ys = [random.uniform(low_bound[yind], up_bound[yind]) for _ in range(n_samples)]

    fs = []
    
    for i in range(len(xs)):
        if args.task == 'MontezumaRevenge-ram-v0':
            obs = np.array([xs[i], ys[i]])
            obs = np.reshape(obs, (1, 2))
            f_value = op.f_function.f_from_features(obs)[0][0]
        elif args.ffunction == 'nns':
            obs = np.array([xs[i], ys[i]])
            obs = np.reshape(obs, (1, 2))
            f_value = op.f_function.f_from_features(obs)[0][0]
        else:
            s = mdp.get_init_state()
            s.data[xind] = xs[i]
            s.data[yind] = ys[i]
            f_value = op.f_function(s)[0][0]            
        fs.append(f_value)

    # TODO: What is the best colormap for all people (including color blinds?) but still appealing for majority?
    #       bwr looks useful, but may be misleading?.
    cmap = matplotlib.cm.get_cmap('plasma')
    normalize = matplotlib.colors.Normalize(vmin=min(fs), vmax=max(fs))
    colors = [cmap(normalize(value)) for value in fs]
    # colors_np = np.asarray(colors)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x=xs, y=ys, c=colors)

    cax, _ = matplotlib.colorbar.make_axes(ax)
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)

    term_th = op.lower_th
    cax.plot([0, 1], [term_th] * 2, 'k')

    term, nonterm = 0, 0
    for f in fs:
        if f < term_th:
            term += 1
        else:
            nonterm += 1
    print(term, 'terms', nonterm, 'nonterms')
    # TODO: Only for pinball domains. What to do for MuJoCo?
    # Obstacles
    if args.tasktype == 'pinball':
        for obs in mdp.domain.environment.obstacles:
            point_list = obs.points
            xlist = []
            ylist = []
            for p in point_list:
                xlist.append(p[0])
                ylist.append(p[1])

            ax.fill(xlist, ylist, 'k')
            
    elif args.task == 'PointMaze-v0' or args.task == 'AntMaze-v0':
        # TODO: (x,y) coordinates start at 0, 0.
        #       How is the coordinates signed?
        maze = [[1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]]
        scale = 8.0
        for y in range(5):
            for x in range(5):
                if maze[y][x] == 1:
                    # We decrement x and y because the (0, 0)-coordinate is set at (1, 1) position in the maze.
                    xbase, ybase = scale * (x - 1), scale * (y - 1)
                    xlist = [xbase, xbase + scale, xbase + scale, xbase]
                    ylist = [ybase, ybase, ybase + scale, ybase + scale]
                    ax.fill(xlist, ylist, 'k')
    elif args.task == 'MontezumaRevenge-ram-v0':
        # TODO: Show the background of the Monte?
        img = imread('./montezuma.jpg')
        ax.imshow(img, zorder=0, extent=[0, 160, 0, 210])
    
    plt.savefig(filename)
    plt.close()
    

# def plot_option(op, args, xind=0, yind=1, filename='visualize_op.pdf'):
#     # Pinball
# 
#     mdp, state_dim, state_bound, num_actions, action_dim, action_bound = get_mdp_params(args)
#     
#     term_x = []
#     term_y = []
#     init_x = []
#     init_y = []
#     nt_x = []
#     nt_y = []
# 
#     n_samples = 2000
#     
#     low_bound = state_bound[0]
#     up_bound = state_bound[1]
# 
#     if np.isinf(low_bound).any() or np.isinf(up_bound).any():
#         bfr = sample_trajectories(mdp, args)
# 
#         ss, _, _, _, _ = bfr.sample(n_samples)
# 
#         max_x = float('-inf')
#         min_x = float('inf')
#         max_y = float('-inf')
#         min_y = float('inf')
#         for i in range(n_samples):
#             x = ss[i].data[xind]
#             y = ss[i].data[yind]
#             max_x = max(x, max_x)
#             min_x = min(x, min_x)
#             max_y = max(y, max_y)
#             min_y = min(y, min_y)
#         low_bound[xind] = min_x
#         up_bound[xind] = max_x
#         low_bound[yind] = min_y
#         up_bound[yind] = max_y
#         
#     for sample in range(n_samples):
#         x = random.uniform(low_bound[xind], up_bound[xind])
#         y = random.uniform(low_bound[yind], up_bound[yind])
#         
#         s = mdp.get_init_state()
#         s.data[xind] = x
#         s.data[yind] = y
#         
#         if op.is_terminal(s):
#             term_x.append(x)
#             term_y.append(y)
#         elif op.is_initiation(s):
#             init_x.append(x)
#             init_y.append(y)
#         else:
#             nt_x.append(x)
#             nt_y.append(y)
# 
#     plt.scatter(x=init_x, y=init_y, c='blue')
#     plt.scatter(x=term_x, y=term_y, c='red')
#     plt.scatter(x=nt_x, y=nt_y, c='gray')
#     plt.savefig(filename)
#     plt.close()

# TODO: add a function to draw out the actual trajectories.

def main(open_plot=True):

    args = arguments()
    
    # Random seeds
    np.random.seed(1234)
    tf.set_random_seed(args.rseed)

    print('task=', args.task)

    mdp, state_dim, state_bound, num_actions, action_dim, action_bound = get_mdp_params(args)



    if args.online:
        # TODO: Think how to solve the restoration for batch normalization.
        op = OptionWrapper(sess=None, experience_buffer=None, obs_dim=state_dim, obs_bound=mdp.bounds(), num_actions=num_actions, action_dim=action_dim, action_bound=action_bound, low_method=args.lowmethod, f_func=args.ffunction, n_units=args.ffuncnunit, init_all=args.initall, restore=True, name='online-option' +  str(args.noptions) + '_' + str(args.ffuncnunit))
        op.restore(args.basedir + '/vis/' + args.task + 'option' + str(args.noptions) + '_' + str(args.ffuncnunit) + '_' + str(args.rseed))
        plot_eigenfunction(op, args, xind=0, yind=1, filename=args.basedir + '/vis/' + args.task + 'online-option' + str(args.noptions) + '_' + str(args.ffuncnunit) + '/' + 'eigenfunc.pdf')
    else:
        op = OptionWrapper(sess=None, experience_buffer=None, obs_dim=state_dim, obs_bound=mdp.bounds(), num_actions=num_actions, action_dim=action_dim, action_bound=action_bound, low_method=args.lowmethod, f_func=args.ffunction, n_units=args.ffuncnunit, init_all=args.initall, restore=True, name='option' +  str(args.noptions) + '_' + str(args.ffuncnunit) + '_' + str(args.rseed))
        op.restore(args.basedir + '/vis/' + args.task + 'option' + str(args.noptions) + '_' + str(args.ffuncnunit) + '_' + str(args.rseed))
        plot_eigenfunction(op, args, xind=0, yind=1, filename=args.basedir + '/vis/' + args.task + 'option' + str(args.noptions) + '_' + str(args.ffuncnunit) + '_' + str(args.rseed) + '/' + 'eigenfunc.pdf')
    
    # plot_option(op, args, xind=0, yind=1, filename='./vis/' + args.task + 'option' + str(args.noptions) + '_' + str(rseed) + '/' + 'vis.pdf')


if __name__ == "__main__":
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.ERROR)
    main(open_plot=not sys.argv[-1] == "no_plot")
