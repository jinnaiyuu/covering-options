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

from util import get_mdp_params

def main(open_plot=True):
    rseed = 1234 # 5678
    # Random seeds
    np.random.seed(1234)
    tf.set_random_seed(rseed)

    parser = argparse.ArgumentParser()

    # pinball files = pinball_box.cfg  pinball_empty.cfg  pinball_hard_single.cfg  pinball_medium.cfg  pinball_simple_single.cfg
    
    # Parameters for the task
    parser.add_argument('--tasktype', type=str, default='pinball')
    parser.add_argument('--task', type=str, default='pinball_empty.cfg')
    parser.add_argument('--base', action='store_true')
    
    parser.add_argument('--nepisodes', type=int, default=100)
    parser.add_argument('--nsteps', type=int, default=200)

    parser.add_argument('--buffersize', type=int, default=512)
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--obuffersize', type=int, default=512)
    parser.add_argument('--obatchsize', type=int, default=128)

    parser.add_argument('--highmethod', type=str, default='linear')
    parser.add_argument('--lowmethod', type=str, default='linear')
    parser.add_argument('--ffunction', type=str, default='fourier')

    
    # Parameters for the Agent
    parser.add_argument('--noptions', type=int, default=5) # (5 = 1 for primitive actions and 4 covering options).

    # Visualization
    parser.add_argument('--render', action='store_true')

    args = parser.parse_args()

    print('tasktype=', args.tasktype)

    state_dim, state_bound, num_actions, action_dim, action_bound = get_mdp_params(args)


    ##############################################
    # Generate the f-function
    ##############################################
    if args.snepisodes == 0:
        op = OptionWrapper(sess=None, experience_buffer=None, obs_dim=state_dim, obs_bound=mdp.bounds(), num_actions=num_actions, action_dim=action_dim, action_bound=action_bound, low_method=args.lowmethod, f_func=args.ffunction, restore=True, name='option' + str(rseed))
        op.restore('./vis/' + args.tasktype + 'option' + str(rseed))
    else:
        bfr = sample_trajectories(mdp, args)
        buf_size = args.snepisodes * args.snsteps
        op = OptionWrapper(sess=None, experience_buffer=bfr, option_b_size=buf_size, obs_dim=state_dim, obs_bound=mdp.bounds(), num_actions=num_actions, action_dim=action_dim, action_bound=action_bound, low_method=args.lowmethod, f_func=args.ffunction, name='option' + str(rseed))
        
        ##############################################
        # Train the option policy
        ##############################################
        buffer_size = args.snsteps * args.snepisodes
        op.train(bfr, batch_size=buffer_size)
    
        op.save('./vis/' + args.tasktype + 'option' + str(rseed))
        exit(0)
    # plot_option(op, './vis/' + 'option' + str(rseed) + '/' + 'vis.pdf')
    ##############################################
    # Evaluate the generated option
    ##############################################
    # print('op.f_function', op.f_function)
    # oagent = OptionAgent(sess=None, obs_dim=state_dim, num_actions=len(mdp.get_actions()), num_options=2, batch_size=1, buffer_size=2, option_batch_size=1, option_buffer_size=2, name='1op')
    # oagent = OptionAgent(sess=None, obs_dim=state_dim, obs_bound=mdp.bounds(), num_actions=num_actions, action_dim=action_dim, action_bound=action_bound, num_options=2, init_all=True, high_method=args.highmethod, low_method=args.lowmethod, f_func=args.ffunction, batch_size=1, buffer_size=2, option_batch_size=1, option_buffer_size=2, name='1op-initall')
    # oagent.add_option(op)
    
    base = OptionAgent(sess=None, obs_dim=state_dim, obs_bound=mdp.bounds(), num_actions=num_actions, action_dim=action_dim, action_bound=action_bound, num_options=1, high_method=args.highmethod, low_method=args.lowmethod, f_func=args.ffunction, batch_size=args.batchsize, buffer_size=args.buffersize, option_batch_size=args.obatchsize, option_buffer_size=args.obuffersize, name='base')
    ddpg = DDPGAgent(sess=None, obs_dim=state_dim, action_dim=action_dim, action_bound=action_bound, buffer_size=args.buffersize, batch_size=args.batchsize, name='ddpg')

    agents = []
    # agents.append(oagent)
    agents.append(base)
    agents.append(ddpg)
    
    mdp.reset()
    
    run_agents_on_mdp(agents, mdp, episodes=args.nepisodes, steps=args.nsteps, instances=args.ninstances, cumulative_plot=True)



if __name__ == "__main__":
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.ERROR)
    main(open_plot=not sys.argv[-1] == "no_plot")
