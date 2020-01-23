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

from util import get_mdp_params, arguments, sample_option_trajectories


def main(open_plot=True):

    args = arguments()

    # Random seeds
    np.random.seed(1234)
    tf.set_random_seed(args.rseed)
    
    print('task=', args.task)

    mdp, state_dim, state_bound, num_actions, action_dim, action_bound = get_mdp_params(args)

    # TODO: Train an option using the trajectories sampled by itself.

    op = OptionWrapper(sess=None, experience_buffer=None, obs_dim=state_dim, obs_bound=mdp.bounds(), num_actions=num_actions, action_dim=action_dim, action_bound=action_bound, low_method=args.lowmethod, f_func=args.ffunction, n_units=args.ffuncnunit, restore=True, init_all=args.initall, reversed_dir=args.reverse, name='option' +  str(args.noptions) + '_' + str(args.ffuncnunit) + '_' + str(args.rseed))

    # if args.reverse:
    #     op.restore('./vis/' + args.task + 'option' + str(args.noptions) + 'rev_' + str(args.ffuncnunit) + '_' + str(args.rseed))
    # else:
    op.restore('./vis/' + args.task + 'option' + str(args.noptions) + '_' + str(args.ffuncnunit) + '_' + str(args.rseed))

    op.reversed_dir = args.reverse

    # TODO: Shouldn't we train the policy based on its own sample frequency?
    if args.restoretraj:
        if args.trajdir == '__default':
            args.trajdir = './vis/' + args.task + 'option' + str(args.noptions - 1) + '_' + str(args.ffuncnunit) + '_' + str(args.rseed) + '/' + 'low_traj'
        
        print('restoring buffer from ' + args.trajdir)
        bfr = ExperienceBuffer()
        # if args.reverse:
        #     bfr.restore('./vis/' + args.task + 'option' + str(args.noptions - 1) + 'rev_' + str(args.rseed) + '/' + 'low_traj')
        # else:
        bfr.restore(args.trajdir)
            
        bfr_size = bfr.size()
        print('bfr_size=', bfr_size) # TODO: parameter?
    else:
        _, bfr = sample_option_trajectories(mdp, args, noptions=args.noptions - 1)
        bfr_size = bfr.size()
        print('bfr_size=', bfr_size)
        
    _, _, r, _, _ = bfr.sample(32)
    print('rewards=', r)

    for _ in range(args.sptrainingstep):
        op.train(bfr, batch_size=min(128, bfr_size))

    if args.reverse:
        op.save('./vis/' + args.task + 'option' + str(args.noptions) + 'rev_' + str(args.ffuncnunit) + '_' + str(args.rseed))
    else:
        op.save('./vis/' + args.task + 'option' + str(args.noptions) + '_' + str(args.ffuncnunit) + '_' + str(args.rseed))

if __name__ == "__main__":
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.ERROR)
    main(open_plot=not sys.argv[-1] == "no_plot")
