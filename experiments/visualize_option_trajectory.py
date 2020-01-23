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


def plot_trajectory(traj, mdp, args, filename='trajectory.pdf'):
    

    if args.tasktype == 'pinball':
        xs = [s.x for s in traj]
        ys = [s.y for s in traj]
        
        plt.plot(xs, ys, 'g')
    
        plt.plot(xs[0], ys[0], 'bo')
        plt.plot(xs[-1], ys[-1], 'ro')


        for obs in mdp.domain.environment.obstacles:
            point_list = obs.points
            xlist = []
            ylist = []
            for p in point_list:
                xlist.append(p[0])
                ylist.append(p[1])
    	
            plt.fill(xlist, ylist, 'k')
    elif args.task == 'PointMaze-v0' or args.task == 'AntMaze-v0':
        xs = [s.data[0] for s in traj]
        ys = [s.data[1] for s in traj]

        print('x =', min(xs), ' to ', max(xs))
        print('y =', min(ys), ' to ', max(ys))
    
        plt.plot(xs, ys, 'g')

        plt.plot(xs[0], ys[0], 'bo')
        plt.plot(xs[-1], ys[-1], 'ro')

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
        feature = Monte()
        xs = [feature.feature(s, 0)[0] for s in traj]
        ys = [feature.feature(s, 0)[1] for s in traj]


        # Duplicate detection
        # print('initpos=', xs[0], ys[0])
        dup = 0
        for i in range(1, len(xs)):
            # print('pos', i, '=', int(xs[i]), int(ys[i]))
            # if int(ys[i]) < 50: # do we have non-suicidal options?
            #     dup = i
            if int(xs[i]) == int(xs[0]) and int(ys[i]) == int(ys[0]):
                dup = i
        print('dup=', dup)
        
        xs = xs[min(dup+1, len(xs)-1):]
        ys = ys[min(dup+1, len(ys)-1):]
        
        plt.plot(xs, ys, 'g')

        plt.plot(xs[0], ys[0], 'bo')
        plt.plot(xs[-1], ys[-1], 'ro')
        
        # TODO: Show the background of the Monte?
        img = imread('./montezuma.jpg')
        plt.imshow(img, zorder=0, extent=[0, 160, 0, 210])
        
    
    plt.savefig(filename)
    plt.close()
    

def main(open_plot=True):

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

        print('sampled')
    # TODO: Print a list of states

    samples = low_bfr.buffer
    
    size = low_bfr.size()

    print('size=', size)

    trajectories = []

    cur_o = None
    for i in range(size):
        # TODO: something wrong is happening in the trajectory. Why?
        s, a, r, s2, t, o = samples[i][0], samples[i][1], samples[i][2], samples[i][3], samples[i][4], samples[i][5]

        # assert(t is False)

        # print('o=', o, ', t=', t)

        if cur_o == args.noptions:
            if o == args.noptions and not t and i != size - 1:
                traj.append(s)
            else:
                # traj.append(s2)
                if args.tasktype == 'pinball':
                    t = [s for s in traj if s.x != 0.2 or s.y != 0.2] # TODO: hack to remove the init state.
                else:
                    t = traj
                # for i, s in enumerate(t):
                    # if 0.01466 <= s.data[0] and s.data[0] <= 0.01467:
                    #     t.remove(s)
                    #     # break
                    # print(s.data[0])
                trajectories.append((i, t))
                    
                cur_o = 0
                traj = []

                # TODO: what is the best way to print these figures out?
                # break
        else:
            if o == args.noptions:
                traj = [s]
                cur_o = args.noptions
                
    for traj in trajectories:
        i = traj[0]
        t = traj[1]
        print(i, ' traj length=', len(t))
        if args.reverse:
            plot_trajectory(t, mdp, args, filename=args.basedir + '/vis/' + args.task + 'option' + str(args.noptions) + 'rev_' + str(args.ffuncnunit) + '_' + str(args.rseed) + '/' + 'traj' + str(i) + '.pdf')
        else:
            plot_trajectory(t, mdp, args, filename=args.basedir + '/vis/' + args.task + 'option' + str(args.noptions) + '_' + str(args.ffuncnunit) + '_' + str(args.rseed) + '/' + 'traj' + str(i) + '.pdf')


if __name__ == "__main__":
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.ERROR)
    main(open_plot=not sys.argv[-1] == "no_plot")
