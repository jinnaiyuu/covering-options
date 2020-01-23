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

from util import get_mdp_params, arguments

def main(open_plot=True):
    # TODO: Accept a set of options instead of just one
    args = arguments()

    # Random seeds
    np.random.seed(args.rseed)
    tf.set_random_seed(args.rseed)

    print('tasktype=', args.tasktype)

    mdp, state_dim, state_bound, num_actions, action_dim, action_bound = get_mdp_params(args)

    oagent = OptionAgent(sess=None, obs_dim=state_dim, obs_bound=state_bound, num_actions=num_actions, action_dim=action_dim, action_bound=action_bound, num_options=1 + args.noptions, init_all=args.initall, high_method=args.highmethod, low_method=args.lowmethod, f_func=args.ffunction, batch_size=args.batchsize, buffer_size=args.buffersize, low_update_freq=args.lowupdatefreq, option_batch_size=args.obatchsize, option_buffer_size=args.obuffersize, high_update_freq=args.highupdatefreq, name='op')
    oagent.reset()

    for nop in range(1, args.noptions + 1):
        op = OptionWrapper(sess=None, experience_buffer=None, obs_dim=state_dim, obs_bound=mdp.bounds(), num_actions=num_actions, action_dim=action_dim, action_bound=action_bound, low_method=args.lowmethod, f_func=args.ffunction, init_all=args.initall, restore=True, name='option' + str(nop) + '_'+ str(args.ffuncnunit) + '_' + str(args.rseed))

        if args.trajdir == '__default':
            if args.reverse:
                opdir = './vis/' + args.task + 'option' + str(nop) + 'rev_' + str(args.ffuncnunit) + '_' + str(args.rseed)
            else:
                opdir = './vis/' + args.task + 'option' + str(nop) + '_' + str(args.ffuncnunit) + '_' + str(args.rseed)                
        else:
            # Only one option can be restored from nonstandard locations
            assert(args.noptions == 1)
            opdir = args.trajdir
        op.restore(opdir)
        print('restored option', opdir)
        # print('upper_th=', op.upper_th)
        oagent.add_option(op)
    
    agents = []
    agents.append(oagent)
    
    if args.base:
        base = OptionAgent(sess=None, obs_dim=state_dim, obs_bound=state_bound, num_actions=num_actions, action_dim=action_dim, action_bound=action_bound, num_options=1, high_method=args.highmethod, low_method=args.lowmethod, f_func=args.ffunction, batch_size=args.batchsize, buffer_size=args.buffersize, low_update_freq=args.lowupdatefreq, option_batch_size=1, option_buffer_size=2, high_update_freq=10000000, init_all=args.initall, name='base')
        agents.append(base)
    
    mdp.reset()

    # TODO: We need to count the number of times the agent reached the goal state.
    #       Because from the cumulative rewards, it is hard to see if the agent is performing as intended.
    #       Possible Solutions: (See the previous works first)
    #         1. Plot the number of times the agent reached the goal.
    #         2. Give a positive reward when it reached the goal
    run_agents_on_mdp(agents, mdp, episodes=args.nepisodes, steps=args.nsteps, instances=args.ninstances, cumulative_plot=True)
    
    options = oagent.options
    for nop in range(1, len(options)):
        if args.trajdir == '__default':
            opdir = './vis/' + args.task + 'option' + str(nop) + '_' + str(args.ffuncnunit) + '_' + str(args.rseed)
        else:
            assert(args.noptions == 1)
            opdir = args.trajdir
        # print('upper=', options[nop].upper_th)
        options[nop].save(opdir + '_trained')

    if args.trajdir == '__default':
        bufdir = './vis/' + args.task + 'option' + str(args.noptions) + '_' + str(args.ffuncnunit) + '_' + str(args.rseed)
    else:
        bufdir = args.trajdir
    oagent.option_buffer.save(bufdir + '_trained' + '/' + 'traj')
    oagent.experience_buffer.save(bufdir + '_trained' + '/' + 'low_traj')
        

if __name__ == "__main__":
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.ERROR)
    main(open_plot=not sys.argv[-1] == "no_plot")
