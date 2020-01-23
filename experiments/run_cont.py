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

# Other imports.
# import srl_example_setup
from simple_rl.agents import RandomContAgent, DDPGAgent, DQNAgent, LinearQAgent
from simple_rl.agents.func_approx.Features import Fourier
from simple_rl.tasks import GymMDP, PinballMDP
from simple_rl.run_experiments import run_agents_on_mdp
from options import OptionAgent

def main(open_plot=True):
    # Random seeds
    np.random.seed(1234)
    tf.set_random_seed(1234)

    parser = argparse.ArgumentParser()

    # pinball files = pinball_box.cfg  pinball_empty.cfg  pinball_hard_single.cfg  pinball_medium.cfg  pinball_simple_single.cfg
    
    # Parameters for the task
    parser.add_argument('--tasktype', type=str, default='pinball')
    parser.add_argument('--task', type=str, default='pinball_empty.cfg')
    parser.add_argument('--base', action='store_true')
    
    parser.add_argument('--nepisodes', type=int, default=10)
    parser.add_argument('--nsteps', type=int, default=100)
    parser.add_argument('--ninstances', type=int, default=1)
    
    # Parameters for the Agent
    parser.add_argument('--highmethod', type=str, default='linear')
    parser.add_argument('--lowmethod', type=str, default='linear')
    parser.add_argument('--ffunction', type=str, default='fourier')
    parser.add_argument('--noptions', type=int, default=5) # (5 = 1 for primitive actions and 4 covering options).
    

    # Visualization
    parser.add_argument('--render', action='store_true')

    args = parser.parse_args()

    print('tasktype=', args.tasktype)
    
    if args.tasktype == 'pinball' or args.tasktype == 'p':
        # TODO: Add parameter for Configuration files by --task argument
        gym_mdp = PinballMDP(cfg=args.task, render=args.render)
        state_dim = 4
    elif args.tasktype == 'atari' or args.tasktype == 'mujoco':
        # Gym MDP        
        gym_mdp = GymMDP(env_name=args.task, render=args.render)
        gym_mdp.env.seed(1234)

        state_dims = gym_mdp.env.observation_space.shape
        state_dim = 1
        for d in state_dims:
            state_dim *= d
        print('state_dim=', state_dim)
    else:
        assert(False)

    # TODO: What should we compare against?
    agents = []
    
    if args.tasktype == 'mujoco':
        action_dim = gym_mdp.env.action_space.shape[0]
        action_bound = gym_mdp.env.action_space.high
        op_agent = OptionAgent(sess=None, obs_dim=state_dim, action_dim=action_dim, action_bound=action_bound, num_options=args.noptions, name='OptionAgent')
        base_agent = DDPGAgent(sess=None, obs_dim=state_dim, action_dim=action_dim, action_bound=action_bound, name='Baseline')
    elif args.tasktype == 'atari':
        num_actions = gym_mdp.env.action_space.n
        print('num_actions=', num_actions)
        op_agent = OptionAgent(sess=None, obs_dim=state_dim, num_actions=num_actions, num_options=args.noptions, name='OptionAgent')
        base_agent = DQNAgent(sess=None, obs_dim=state_dim, num_actions=num_actions, name='Baseline')
    elif args.tasktype == 'pinball' or args.tasktype == 'p':
        num_actions = 5
        low_bound, up_bound = mdp.bounds()
        feature = Fourier(state_dim=state_dim, state_up_bound=up_bound, state_low_bound=low_bound, order=4)
        base_agent = LinearQAgent(actions=gym_mdp.get_actions(), feature=feature, sarsa=False, name='baseline')
        op_agent = OptionAgent(sess=None, obs_dim=state_dim, num_actions=num_actions, num_options=args.noptions, high_method='linear', low_method='linear', name='OptionAgent')
        # base_agent = DQNAgent(sess=None, obs_dim=state_dim, num_actions=num_actions, name='Baseline')
    else:
        assert(False)
        
    if args.base:
        agents.append(base_agent)
    else:
        agents.append(op_agent)
        
    run_agents_on_mdp(agents, gym_mdp, episodes=args.nepisodes, steps=args.nsteps, verbose=True, instances=args.ninstances, cumulative_plot=False, open_plot=False)

if __name__ == "__main__":
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.ERROR)
    main(open_plot=not sys.argv[-1] == "no_plot")
