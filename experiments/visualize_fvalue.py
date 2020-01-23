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


def plot_fvalue(bfr, op, filename='trajectory.pdf'):

    s = [bfr.buffer[i][0] for i in range(bfr.size())]

    xs = range(len(s))

    ys = [op.f_function(i)[0][0] for i in s]

    os = [bfr.buffer[i][5] for i in range(bfr.size())] # Option of the action

    lb = op.lower_th
    

    plt.scatter(xs, ys, c=os)

    print('threshold=', lb)
    plt.axhline(y=lb)
    
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

    
    size = low_bfr.size()

    op = OptionWrapper(sess=None, experience_buffer=None, obs_dim=state_dim, obs_bound=mdp.bounds(), num_actions=num_actions, action_dim=action_dim, action_bound=action_bound, low_method=args.lowmethod, f_func=args.ffunction, n_units=args.ffuncnunit, init_all=args.initall, restore=True, name='option' +  str(args.noptions) + '_' + str(args.ffuncnunit) + '_' + str(args.rseed))
    op.restore(args.basedir + '/vis/' + args.task + 'option' + str(args.noptions) + '_' + str(args.ffuncnunit) + '_' + str(args.rseed))
    
    filename = args.basedir + '/vis/' + args.task + 'option' + str(args.noptions) + '_' + str(args.ffuncnunit) + '_' + str(args.rseed) + '/' + 'fvalues.pdf'
    
    plot_fvalue(low_bfr, op, filename=filename)


if __name__ == "__main__":
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.ERROR)
    main(open_plot=not sys.argv[-1] == "no_plot")
