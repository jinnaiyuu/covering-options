#!/usr/bin/env python
# Python imports.
import sys
import logging
import argparse
import numpy as np
import tensorflow as tf
import argparse
import random

from util import arguments, sample_option_trajectories, get_mdp_params
from simple_rl.agents.func_approx.ExperienceBuffer import ExperienceBuffer


def main():
    args = arguments()

    mdp, state_dim, state_bound, num_actions, action_dim, action_bound = get_mdp_params(args)

    bfr, low_bfr = sample_option_trajectories(mdp, args, noptions=args.noptions)

    # TODO: Trajectories are generated using noptions-1 options.

    if args.reverse:    
        bfr.save(args.basedir + '/vis/' + args.task + 'option' + str(args.noptions) + 'rev_' + str(args.ffuncnunit) + '_' + str(args.rseed) + '/' + 'traj')
        low_bfr.save(args.basedir + '/vis/' + args.task + 'option' + str(args.noptions) + 'rev_' + str(args.ffuncnunit) + '_' + str(args.rseed) + '/' + 'low_traj')
    else:
        bfr.save(args.basedir + '/vis/' + args.task + 'option' + str(args.noptions) + '_' + str(args.ffuncnunit) + '_' + str(args.rseed) + '/' + 'traj')
        low_bfr.save(args.basedir + '/vis/' + args.task + 'option' + str(args.noptions) + '_' + str(args.ffuncnunit) + '_' + str(args.rseed) + '/' + 'low_traj')


    print('bfr  size=', bfr.size())
    print('lbfr size=', low_bfr.size())

    if args.task == 'PointMaze-v0':
        s, a, r, s, t = low_bfr.sample(20)
        for state in s:
            # print('s=', state) # TODO: how do we get the X, Y coordinates of the agent?
            print('x,y=', state.data[0], state.data[1])

    if args.task == 'MontezumaRevenge-ram-v0':
        s, a, r, s, t = low_bfr.sample(20)
        def getByte(ram, row, col):
            row = int(row, 16) - 8
            col = int(col, 16)
            return ram[row*16+col]
        for state in s:
            x = int(getByte(state.data, 'a', 'a'))
            y = int(getByte(state.data, 'a', 'b'))
            
            x_img = int(210.0 * (float(x) - 1) / float((9 * 16 + 8) - 1))
            y_img = int(160.0 * (float(y) - (8 * 16 + 6)) / float((15 * 16 + 15) - (8 * 16 + 6)))

            print('(ram) x, y =', x, y)
            print('(img) x, y =', x_img, y_img)
            
    # bfr2 = ExperienceBuffer()

    # bfr2.restore('./vis/' + args.task + 'option' + str(args.noptions) + '_' + str(args.rseed) + '/' + 'traj')

    # print('bfr2 size=', bfr2.size())

if __name__ == "__main__":
    main()
