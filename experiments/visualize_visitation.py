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
from simple_rl.agents.func_approx.Features import Fourier, Monte
from simple_rl.tasks import GymMDP, PinballMDP
from simple_rl.tasks.pinball.PinballStateClass import PinballState
from simple_rl.agents.func_approx.ExperienceBuffer import ExperienceBuffer

from simple_rl.run_experiments import run_agents_on_mdp

from options.OptionWrapper import OptionWrapper
from options.OptionAgent import OptionAgent

from util import get_mdp_params, arguments, sample_option_trajectories


def plot_visitation(traj, mdp, args, filename='visitation.pdf'):
    # TODO: it only works for Pinball.

    if args.tasktype == 'pinball':
        xs = [s.data[0] for s in traj]
        ys = [s.data[1] for s in traj]
        
        plt.plot(xs, ys, 'b.', alpha=0.5)
        for obs in mdp.domain.environment.obstacles:
            point_list = obs.points
            xlist = []
            ylist = []
            for p in point_list:
                xlist.append(p[0])
                ylist.append(p[1])
                
            plt.fill(xlist, ylist, 'k')
    elif args.task == 'PointMaze-v0' or args.task == 'AntMaze-v0':
        xs = []
        ys = []
        for s in traj:
            if -4.0 <= s.data[0] and s.data[0] <= -4.0 + 8.0 * 3.0 and \
               -4.0 <= s.data[1] and s.data[1] <= -4.0 + 8.0 * 3.0:
                xs.append(s.data[0])
                ys.append(s.data[1])
        
        plt.plot(xs, ys, 'b.', alpha=0.5)
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
                    xbase, ybase = scale * (x - 1.5), scale * (y - 1.5)
                    xlist = [xbase, xbase + scale, xbase + scale, xbase]
                    ylist = [ybase, ybase, ybase + scale, ybase + scale]
                    plt.fill(xlist, ylist, 'k')
    elif args.task == 'MontezumaRevenge-ram-v0':
        # TODO: Show the background of the Monte?
        img = imread('./montezuma.jpg')
        plt.imshow(img, zorder=0, extent=[0, 160, 0, 210])

        feature = Monte()
                
        pairs = [feature.feature(s, 0) for s in traj]
        xs_img = [p[0] for p in pairs]
        ys_img = [p[1] for p in pairs]
        
        plt.xlim([0, 160])
        plt.ylim([0, 210])
        plt.plot(xs_img, ys_img, 'r.', alpha=0.5)
    
    plt.savefig(filename)
    plt.close()

    
def main(open_plot=True):
    # TODO: Refactor and combine visualize_visitation, visualize_option, visualize_option_trajectory?
    
    # Plot the visitation statistics 

    args = arguments()
    
    # Random seeds
    np.random.seed(1234)
    tf.set_random_seed(args.rseed)

    print('task=', args.task)

    mdp, state_dim, state_bound, num_actions, action_dim, action_bound = get_mdp_params(args)


    if args.restoretraj:
        # bfr = ExperienceBuffer()
        # bfr.restore('./vis/' + args.task + 'option' + str(args.noptions) + '_' + str(args.rseed) + '/' + 'traj')
        low_bfr = ExperienceBuffer()
        if args.reverse:
            low_bfr.restore(args.basedir + '/vis/' + args.task + 'option' + str(args.noptions) + 'rev_' + str(args.ffuncnunit) + '_' + str(args.rseed) + '/' + 'low_traj')
        else:
            low_bfr.restore(args.basedir + '/vis/' + args.task + 'option' + str(args.noptions) + '_' + str(args.ffuncnunit) + '_' + str(args.rseed) + '/' + 'low_traj')
    else:
        _, low_bfr = sample_option_trajectories(mdp, args, noptions=args.noptions)


    # TODO: Print a list of states

    samples = low_bfr.buffer
    
    size = low_bfr.size()
    
    cur_o = None
    traj = [samples[i][0] for i in range(size)]

    if args.reverse:
        plot_visitation(traj, mdp, args, filename=args.basedir + '/vis/' + args.task + 'option' + str(args.noptions) + 'rev_' + str(args.ffuncnunit) + '_' + str(args.rseed) + '/' + 'visitations' + '.pdf')
    else:
        plot_visitation(traj, mdp, args, filename=args.basedir + '/vis/' + args.task + 'option' + str(args.noptions) + '_' + str(args.ffuncnunit) + '_' + str(args.rseed) + '/' + 'visitations' + '.pdf')
    


if __name__ == "__main__":
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.ERROR)
    main(open_plot=not sys.argv[-1] == "no_plot")
