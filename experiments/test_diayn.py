#!/usr/bin/env python
# Python imports.
import sys
import logging
import argparse
import numpy as np
import tensorflow as tf
import argparse
import random

from simple_rl.agents.func_approx.ExperienceBuffer import ExperienceBuffer
from simple_rl.agents import DiaynAgent
from simple_rl.run_experiments import run_agents_on_mdp

from util import arguments, sample_option_trajectories, get_mdp_params
from options.OptionWrapper import DiaynOption
from options.OptionAgent import OptionAgent


def save(args):

    mdp, obs_dim, state_bound, num_actions, action_dim, action_bound = get_mdp_params(args)
    
    agent = DiaynAgent(sess=None, obs_dim=obs_dim, num_actions=num_actions, num_options=args.noptions, action_dim=action_dim, action_bound=action_bound, batch_size=32, update_freq=32, alpha=1.0)

    agent.set_diversity(True)

    run_agents_on_mdp([agent], mdp, episodes=args.snepisodes, steps=args.snsteps, instances=1, cumulative_plot=True)


    if args.trajdir == '__default':
        prefix = '.'
    else:
        prefix = args.trajdir
        
    agent.save(directory=prefix + '/vis' + '/' + str(args.task) + 'option' + str(args.noptions) + 'diayn', name='diayn-pretrain')

def restore(args):

    mdp, obs_dim, state_bound, num_actions, action_dim, action_bound = get_mdp_params(args)

    rst = DiaynAgent(sess=None, obs_dim=obs_dim, num_actions=num_actions, action_dim=action_dim, action_bound=action_bound, num_options=args.noptions, batch_size=1, update_freq=1, alpha=1.0)
    rst.restore(directory=prefix + '/vis' + '/' + str(args.task) + 'option' + str(args.noptions) + 'diayn', name='diayn-pretrain')

    rst.set_diversity(False)

    oagent = OptionAgent(sess=None, obs_dim=obs_dim, obs_bound=state_bound, num_actions=num_actions, action_dim=action_dim, action_bound=action_bound, num_options=1 + args.noptions, init_all=args.initall, high_method=args.highmethod, low_method=args.lowmethod, f_func=args.ffunction, batch_size=args.batchsize, buffer_size=args.buffersize, low_update_freq=args.lowupdatefreq, option_batch_size=args.obatchsize, option_buffer_size=args.obuffersize, high_update_freq=args.highupdatefreq, name='diayn' + str(args.noptions))
    
    for i in range(args.noptions):
        op = DiaynOption(rst, i, args.termprob)
        oagent.add_option(op)
    
    run_agents_on_mdp([oagent], mdp, episodes=args.nepisodes, steps=args.nsteps, instances=args.ninstances, cumulative_plot=True)
    

if __name__ == "__main__":
    args = arguments()
    if args.exp == 'sample':
        save(args)
    elif args.exp == 'evaloff':
        restore(args)
    else:
        print('set --exp sample or evaloff')
        assert(False)
