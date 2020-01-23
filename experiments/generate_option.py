#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Python imports.
import sys
import logging
import numpy as np
import tensorflow as tf
import argparse
import random

# Other imports.
# import srl_example_setup
from simple_rl.agents import RandomAgent, DDPGAgent, DQNAgent, LinearQAgent
from simple_rl.agents.func_approx.Features import Fourier
from simple_rl.tasks import GymMDP, PinballMDP
from simple_rl.tasks.pinball.PinballStateClass import PinballState
from simple_rl.agents.func_approx.ExperienceBuffer import ExperienceBuffer

from simple_rl.run_experiments import run_agents_on_mdp

from options.OptionWrapper import OptionWrapper
from options.OptionAgent import OptionAgent

from util import sample_option_trajectories, get_mdp_params, arguments

def main(open_plot=True):
    # TODO: Accept set of options and generate a new option based on them.
    
    args = arguments()

    np.random.seed(1234)
    # tf.set_random_seed(args.rseed)
    # tf.set_random_seed(5678)
    # tf.set_random_seed(5408)
    tf.set_random_seed(2345)
    
    print('tasktype=', args.tasktype)


    mdp, state_dim, state_bound, num_actions, action_dim, action_bound = get_mdp_params(args)

    # We generate k-th option based on the previous k-1 options.

    if args.restoretraj:
        bfr = ExperienceBuffer()
        if args.reverse:
            print('restoring buffer from ' + './vis/' + args.task + 'option' + str(args.noptions-1) + 'rev_' + str(args.ffuncnunit) + '_' + str(args.rseed) + '/' + 'traj')
            bfr.restore(args.basedir + '/vis/' + args.task + 'option' + str(args.noptions - 1) + 'rev_' + str(args.ffuncnunit) + '_' + str(args.rseed) + '/' + 'traj')
        else:
            print('restoring buffer from ' + './vis/' + args.task + 'option' + str(args.noptions-1) + '_' + str(args.ffuncnunit) + '_' + str(args.rseed) + '/' + 'traj')
            bfr.restore(args.basedir + '/vis/' + args.task + 'option' + str(args.noptions - 1) + '_' + str(args.ffuncnunit) + '_' + str(args.rseed) + '/' + 'traj')
        bfr_size = bfr.size()
        print('bfr_size=', bfr_size) # TODO: parameter?
    else:
        bfr, _ = sample_option_trajectories(mdp, args, noptions=args.noptions - 1)
        bfr_size = bfr.size()
        print('bfr_size=', bfr_size)

    # TODO: In graph theory, inserting an edge results in significant change to the topology.
    #       However, seems adding just one transition sample to the NN does not change it too much.
    #       Can we tackle this problem other than sampling the trajectories again?

    
    op = OptionWrapper(sess=None, experience_buffer=bfr, option_b_size=min(32, bfr_size), sp_training_steps=args.sptrainingstep, obs_dim=state_dim, obs_bound=state_bound, num_actions=num_actions, action_dim=action_dim, action_bound=action_bound, low_method=args.lowmethod, f_func=args.ffunction, n_units=args.ffuncnunit, init_all=args.initall, restore=None, reversed_dir=args.reverse, name='option' + str(args.noptions) + '_' + str(args.ffuncnunit) + '_' + str(args.rseed))

    # if args.train:
    #     op.train(bfr, batch_size=args.snepisodes * args.snsteps)

    if args.reverse:
        filename = args.basedir + '/vis/' + args.task + 'option' + str(args.noptions) + 'rev_' + str(args.ffuncnunit) + "_" + str(args.rseed)
    else:
        filename = args.basedir + '/vis/' + args.task + 'option' + str(args.noptions) + '_'    + str(args.ffuncnunit) + '_' + str(args.rseed)
        
    op.save(filename)

if __name__ == "__main__":
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.ERROR)
    main(open_plot=not sys.argv[-1] == "no_plot")
