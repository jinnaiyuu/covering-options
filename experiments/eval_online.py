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
    args = arguments()

    # Random seeds
    np.random.seed(args.rseed)
    tf.set_random_seed(args.rseed)

    print('tasktype=', args.tasktype)

    mdp, state_dim, state_bound, num_actions, action_dim, action_bound = get_mdp_params(args)

    oagent = OptionAgent(sess=None, obs_dim=state_dim, obs_bound=state_bound, num_actions=num_actions, action_dim=action_dim, action_bound=action_bound, num_options=args.noptions, init_all=args.initall, high_method=args.highmethod, low_method=args.lowmethod, f_func=args.ffunction, batch_size=args.batchsize, buffer_size=args.buffersize, option_batch_size=args.obatchsize, option_buffer_size=args.obuffersize, option_freq=args.ofreq, option_min_steps=args.ominsteps, name=str(args.noptions) + 'op-initall')
    

    agents = []
    agents.append(oagent)

    if args.base:
        base = OptionAgent(sess=None, obs_dim=state_dim, obs_bound=state_bound, num_actions=num_actions, action_dim=action_dim, action_bound=action_bound, num_options=1, high_method=args.highmethod, low_method=args.lowmethod, f_func=args.ffunction, batch_size=args.batchsize, buffer_size=args.buffersize, option_batch_size=1, option_buffer_size=2, init_all=args.initall, name='base')
        agents.append(base)
    
    mdp.reset()
    
    run_agents_on_mdp(agents, mdp, episodes=args.nepisodes, steps=args.nsteps, instances=args.ninstances, cumulative_plot=True)

    # TODO: Save the options learned by the agent
    options = oagent.generated_options[1]
    print('options=', options)
    for i, op in enumerate(options):
        if i == 0:
            continue
        op.save('./vis/' + args.task + 'online-option' + str(i) + '_' + str(args.rseed))
        # op.save('./vis/' + args.task + 'online-option' + str(i) + '_' + str(args.rseed))
        

if __name__ == "__main__":
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.ERROR)
    main(open_plot=not sys.argv[-1] == "no_plot")
