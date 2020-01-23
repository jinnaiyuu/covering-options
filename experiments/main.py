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

from scipy.misc import imread, imsave

# Other imports.
# import srl_example_setup
from simple_rl.agents import GenerateRandomAgent, DDPGAgent, DQNAgent, LinearQAgent
from simple_rl.agents.func_approx.Features import Fourier, Monte
from simple_rl.tasks import GymMDP, PinballMDP
from simple_rl.tasks.pinball.PinballStateClass import PinballState
from simple_rl.agents.func_approx.ExperienceBuffer import ExperienceBuffer

from simple_rl.run_experiments import run_agents_on_mdp

from options.OptionWrapper import OptionWrapper, CoveringOption
from options.OptionAgent import OptionAgent

import util
from util import get_mdp_params, arguments, sample_option_trajectories


def plot_bg(mdp, args, fig, ax):
    if args.tasktype == 'pinball':
        for obs in mdp.domain.environment.obstacles:
            point_list = obs.points
            xlist = []
            ylist = []
            for p in point_list:
                xlist.append(p[0])
                ylist.append(p[1])
                
            ax.fill(xlist, ylist, 'k')
    elif 'Point' in args.task or 'Ant' in args.task:
        maze = mdp.env.MAZE_STRUCTURE

        scale = mdp.env.MAZE_SIZE_SCALING
        
        for y in range(len(maze)):
            for x in range(len(maze[0])):
                if maze[y][x] == 'r':
                    apos = (x, y)
                    break
        for y in range(len(maze)):
            for x in range(len(maze[0])):
                if maze[y][x] == 1:
                    # We decrement x and y because the (0, 0)-coordinate is set at (1, 1) position in the maze.
                    xbase, ybase = scale * (x - apos[0] - 0.5), scale * (y - apos[1] - 0.5)
                    xlist = [xbase, xbase + scale, xbase + scale, xbase]
                    ylist = [ybase, ybase, ybase + scale, ybase + scale]
                    ax.fill(xlist, ylist, 'k')
    elif args.tasktype == 'atariram':
        if args.task == 'MontezumaRevenge-ram-v0':
            img = imread('./montezuma.jpg')
        elif args.task == 'Freeway-ram-v0':
            img = imread('./freeway.png')
        elif args.task == 'MsPacman-ram-v0':
            img = imread('./ms_pacman.png')
        else:
            print('background not available')
            assert(False)
        ax.imshow(img, zorder=0, extent=[0, 160, 0, 210])
    else:
        print('No background image provided')
    

def plot_op(op, args, mdp, state_bound, filename):
    print('visop')
    n_samples = 2000

    # TODO: Visualize the options according to the direction
    # direction = args.reverse

    if args.restoretraj:
        sample, _, _, _, _ = bfr.sample(n_samples)

        if args.task == 'MontezumaRevenge-ram-v0':
            feature = Monte()
            xs = [feature.feature(s, 0)[0] for s in samples]
            ys = [feature.feature(s, 0)[1] for s in samples]
        elif args.ffunction == 'nns':
            feature = Subset(state_dim, [0, 1])
            xs = [feature.feature(s, 0)[0] for s in samples]
            ys = [feature.feature(s, 0)[1] for s in samples]        
        else:
            xs = [s.data[0] for s in samples]
            ys = [s.data[1] for s in samples]
    else:
        # TODO: bounds should be implemented inside the tasks.
        if 'Ant' in args.task or 'Point' in args.task:
            up_bound_x, low_bound_x, up_bound_y, low_bound_y = util.bounds(mdp)
            # low_bound_x = - 0.5
            # low_bound_y = -4.0
            # up_bound_x = -4.0 + 8.0 * 3.0
            # up_bound_y = -4.0 + 8.0 * 3.0
        elif args.tasktype == 'atari':
            low_bound_x = 0.0
            low_bound_y = 0.0
            up_bound_x = 160.0
            up_bound_y = 210.0
        else:
            low_bound_x = state_bound[0][0]
            low_bound_y = state_bound[0][1]
            up_bound_x = state_bound[1][0]
            up_bound_y = state_bound[1][1]
        xs = [random.uniform(low_bound_x, up_bound_x) for _ in range(n_samples)]
        ys = [random.uniform(low_bound_y, up_bound_y) for _ in range(n_samples)]
    fs = []
    for i in range(len(xs)):
        if args.task == 'MontezumaRevenge-ram-v0':
            obs = np.array([xs[i], ys[i]])
            obs = np.reshape(obs, (1, 2))
            f_value = op.f_function.f_from_features(obs)[0][0]
        elif args.ffunction == 'nns':
            obs = np.array([xs[i], ys[i]])
            obs = np.reshape(obs, (1, 2))
            f_value = op.f_function.f_from_features(obs)[0][0]
        else:
            s = mdp.get_init_state()
            s.data[0] = xs[i]
            s.data[1] = ys[i]
            f_value = op.f_function(s)[0][0]            
        fs.append(f_value)

    # TODO: Find the best color mapping for visualization.
    # TODO: What is the best thing we can do for color blinds? Intensity of the plot?

    # TODO:
    # if args.reverse:

    cmap = matplotlib.cm.get_cmap('Blues')
    # cmap = matplotlib.cm.get_cmap('plasma')

    
    if args.reverse:
        term_th = op.upper_th
        normalize = matplotlib.colors.Normalize(vmin=min(fs), vmax=term_th) # TODO: Does this give us an inverse direction?
        
        colors = []
        for value in fs:
            if value < term_th:
                # colors.append(cmap(1.0 - normalize(value)))
                colors.append(cmap(normalize(value)))
            else:
                # TODO: What is gray rgb?
                # colors.append((0.15, 0.15, 0.15))
                colors.append((0.15, 0.15, 0.15))
                # colors.append((0.0 , 0.0 , 1.0))
        
    else:
        term_th = op.lower_th    
        normalize = matplotlib.colors.Normalize(vmin=term_th, vmax=max(fs))
    
        colors = []
        for value in fs:
            if value > term_th:
                colors.append(cmap(1.0 - normalize(value)))
            else:
                colors.append((0.15, 0.15, 0.15))
                # colors.append((0, 0, 0))
    # print('colors=', colors)
    # colors_np = np.asarray(colors)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x=xs, y=ys, c=colors)

    cax, _ = matplotlib.colorbar.make_axes(ax)
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)

    # cax.plot([0, 1], [term_th] * 2, 'k')

    term, nonterm = 0, 0
    for f in fs:
        if f < term_th:
            term += 1
        else:
            nonterm += 1
    print(term, 'terms', nonterm, 'nonterms')

    plot_bg(mdp, args, fig, ax)

    plt.savefig(filename, bbox_inches = 'tight', pad_inches = 0)
    plt.close()

def plot_vis(traj, args, mdp, filename):
    if args.tasktype == 'pinball':
        xs = [s.data[0] for s in traj]
        ys = [s.data[1] for s in traj]
    elif 'Point' in args.task or 'Ant' in args.task:
        xs = []
        ys = []

        up_bound_x, low_bound_x, up_bound_y, low_bound_y = util.bounds(mdp)
        for s in traj:
            if low_bound_x <= s.data[0] and s.data[0] <= up_bound_x and \
               low_bound_y<= s.data[1] and s.data[1] <= up_bound_y:
                xs.append(s.data[0])
                ys.append(s.data[1])        
    elif args.task == 'MontezumaRevenge-ram-v0':
        feature = Monte()
        pairs = [feature.feature(s, 0) for s in traj]
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        
        plt.xlim([0, 160])
        plt.ylim([0, 210])
    elif args.tasktype == 'atari':
        print('not implemented yet')
        assert(False)
    elif args.tasktype == 'atariram':
        if args.task == 'Freeway-ram-v0' or args.task == 'MsPacman-ram-v0':
            if args.task == 'Freeway-ram-v0':
                target = (252, 252, 84) # Chicken
            elif args.task == 'MsPacman-ram-v0':
                target = (210, 164, 74) # Color of the pacman.
            else:
                assert(False)
            xs = []
            ys = []

            for s in traj:
                fig = np.asarray(s.data)
                shape = fig.shape
                # print('shape=', shape)
                pos = None
                for x in range(shape[1]):
                    for y in range(shape[0]):
                        if tuple(fig[y][x]) == target:
                            pos = (x, 210 - y)
                            break
                    if pos is not None:
                        break
                if pos is not None:
                    xs.append(pos[0])
                    ys.append(pos[1])
        plt.xlim([0, 160])
        plt.ylim([0, 210])
    else:
        assert(False)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    plot_bg(mdp, args, fig, ax)
    
    ax.plot(xs, ys, 'b.', alpha=0.5) # Red is better for visualizing in Monte
    

    # init_s = mdp.domain.s0()[0]
    # print('init_s=', init_s,)
    # ax.plot(init_s[0], init_s[1], 'bo')
    # goal = mdp.domain.environment.target_pos
    # ax.plot(goal[0], goal[1], 'rx')
    # print('goal=', goal)
    
    plt.savefig(filename + ".pdf", bbox_inches = 'tight', pad_inches = 0)
    plt.savefig(filename + ".png", bbox_inches = 'tight', pad_inches = 0)
    plt.close()


def plot_traj(traj, mdp, args, filename='trajectory'):
    fig, ax = plt.subplots(figsize=(8, 6))
    if args.tasktype == 'pinball':
        xs = [s.x for s in traj]
        ys = [s.y for s in traj]
        
    elif 'Point' in args.task or 'Ant' in args.task:
        # xs = [s.data[0] for s in traj]
        # ys = [s.data[1] for s in traj]
        xs = [maze_width(mdp) - s.data[1] for s in traj]
        ys = [maze_height(mdp) -s.data[0] for s in traj]

        print('x =', min(xs), ' to ', max(xs))
        print('y =', min(ys), ' to ', max(ys))
        
        import time 
        for i in range(0, int(len(xs) / 3)+1):
            # Plot every three states in a trajectory (otherwise it gets unreadable)
            mdp.env.wrapped_env.set_xy([xs[i*3], ys[i*3]])
            mdp.env.wrapped_env.set_ori(traj[i*3].data[2])
            mdp.env.step([0, 0])
            time.sleep(0.1)
            img = mdp.env.render(mode='rgb_array')

            if i < 10:
                imsave(args.task + '_0' + str(i) + '.png', img)
            else:
                imsave(args.task + '_' + str(i) + '.png', img)

    elif args.task == 'MontezumaRevenge-ram-v0':
        feature = Monte()
        xs = [feature.feature(s, 0)[0] for s in traj]
        ys = [feature.feature(s, 0)[1] for s in traj]
        # Duplicate detection
        dup = 0
        for i in range(1, len(xs)):
            if int(xs[i]) == int(xs[0]) and int(ys[i]) == int(ys[0]):
                dup = i
        print('dup=', dup)
        
        xs = xs[min(dup+1, len(xs)-1):]
        ys = ys[min(dup+1, len(ys)-1):]
    elif args.tasktype == 'atari':
        init_s = traj[0]
        fig = np.reshape(init_s.data, (105, 80, 3))
        # fig = np.reshape(init_s.data, (210, 160, 3))
        plt.imshow(fig, vmin=0, vmax=255)
        plt.savefig(filename + '_init.pdf', bbox_inches = 'tight', pad_inches = 0)
        goal_s = traj[-1]
        fig = np.reshape(goal_s.data, (105, 80, 3))
        plt.imshow(fig, vmin=0, vmax=255)
        plt.savefig(filename + '_goal.pdf', bbox_inches = 'tight', pad_inches = 0)
        return
    elif args.tasktype == 'atariram':
        if args.task == "MsPacman-ram-v0":
            target = (210, 164, 74) # Color of the pacman.
        elif args.task == "Freeway-ram-v0":
            target = (252, 252, 84) # Chicken
        else:
            for s in traj:
                fig = np.asarray(s.data)
                shape = fig.shape
                colors = set()
                pos = None
                for x in range(shape[1]):
                    for y in range(shape[0]):
                        if tuple(fig[y][x]) not in colors:
                            colors.add(tuple(fig[y][x]))
                            pos.append((x, y) + tuple(fig[y][x]))

                print('pos=', pos)
            assert(False)

        xs = []
        ys = []

        for s in traj:
            fig = np.asarray(s.data)
            shape = fig.shape
            pos = None
            for x in range(shape[1]):
                for y in range(shape[0]):
                    if tuple(fig[y][x]) == target:
                        pos = (x, 210 - y)
                        break
                    if pos is not None:
                        break
            if pos is not None:
                xs.append(pos[0])
                ys.append(pos[1])
    else:
        print('bg Not implemented')
    
    plot_bg(mdp, args, fig, ax)
        
    plt.plot(xs, ys, 'g')
        
    plt.plot(xs[0], ys[0], 'bo')
    plt.plot(xs[-1], ys[-1], 'ro')
    
    plt.savefig(filename + '_' + str(len(traj)) + '.pdf', bbox_inches = 'tight', pad_inches = 0)
    plt.close()

def plot_terms(goals, mdp, args, filename='terms.pdf'):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if args.tasktype == 'atariram' or args.tasktype == 'atari':

        if "MsPacman" in args.task:
            target = (210, 164, 74) # Color of the pacman.
            bg = imread('./ms_pacman.png')
        elif "Freeway" in args.task:
            target = (252, 252, 84)
            bg = imread('./freeway.png')
        elif "Amidar" in args.task:
            # target = None
            target = (66, 158, 130)
            bg = imread('./amidar.png')
        elif "MontezumaRevenge" in args.task:
            target = (228, 111, 111)
            bg = imread('./montezuma.jpg')
        elif "BankHeist" in args.task:
            target = (223, 183, 85)
            bg = imread('./bank_heist.png')
        else:
            assert(False)

        if target is not None:
            ax.imshow(bg, zorder=0, extent=[0, 160, 0, 210], alpha=0.5)

        posx = []
        posy = []

        for numgoal, s in enumerate(goals):
            # TODO: Should we subtract the background?
            img = np.asarray(s[0].data)
            if args.tasktype == 'atari':
                img = np.reshape(img, (105, 80, 3))
                # print('img=', img)
                # ax.imshow(img, zorder=0, extent=[0, 80, 0, 105], alpha=0.5)
                # plt.savefig(filename + '_test_terms.pdf', bbox_inches = 'tight', pad_inches = 0)
            shape = img.shape
            pos = None

            # print('shape=', shape)

            if target is None:
                # List colors.
                colors = set()
                pos = []
                for x in range(shape[1]):
                    for y in range(shape[0]):
                        if tuple(img[y][x]) not in colors and (25 < 105-y) and (105-y < 76):                            
                            colors.add(tuple(img[y][x]))
                            pos.append((x, y) + tuple(img[y][x]))
                            posx.append(x)
                            posy.append(105 - y)
                            ax.imshow(img, zorder=0, extent=[0, 80, 0, 105], alpha=0.5)
                            plt.plot(posx, posy, 'bo', markersize=16)
                            plt.savefig(filename + '_' + str(numgoal) + 'goal_' + str(x) + '-' + str(y) + '--' + str(tuple(img[y][x])) + '_terms.pdf', bbox_inches = 'tight', pad_inches = 0)
                            break
                continue
            else:
                for x in range(shape[1]):
                    for y in range(shape[0]):
                        if tuple(img[y][x]) == target:
                            # HACK: The color of the agent is the same as that of the background color. To avoid detecting the bg.
                            if args.task == 'Freeway-ram-v0' and 75 < y and y < 125:
                                continue
                            if args.task == 'Amidar-ram-v0' and 210 - y > 48:
                                continue
                            if args.task == 'BankHeist-v0' and (25 < 105-y) and (105-y < 75):
                                continue
                            pos = (x*2, 210 - y*2)
                            break
                    if pos is not None:
                        break
                if pos is not None:
                    posx.append(pos[0])
                    posy.append(pos[1])
        plt.plot(posx, posy, 'o', color='red', markersize=16)
        # plt.plot(posx, posy, 'ro', markersize=16)
        plt.savefig(filename + '_terms.pdf', bbox_inches = 'tight', pad_inches = 0)
            
    

    
def main(open_plot=True):
    
    args = arguments()
    
    # Random seeds
    np.random.seed(args.rseed)
    tf.set_random_seed(args.rseed)

    print('task=', args.task)

    mdp, state_dim, state_bound, num_actions, action_dim, action_bound = get_mdp_params(args)

    #################################
    # 1. Retrieve trajectories
    #################################
    if args.trajdir == '__default':
        prefix = '.'
    else:
        prefix = args.trajdir

        
    if args.exp == "generate" or args.exp == "train":
        pathnop = str(args.noptions - 1)
    else:
        pathnop = str(args.noptions)
        
    # if args.reverse:
    #     dirop = 'rev_'
    # else:
    #     dirop = '_'
    dirop = '_'

    # pathdir: directory for the trajectories
    # opdir  : directory for the option
    pathdir = prefix + '/vis/' + args.task + 'option' + pathnop + dirop + str(args.ffuncnunit) + '_' + str(args.rseed)

    opdir = prefix + '/vis/' + args.task + 'option' + str(args.noptions) + dirop + str(args.ffuncnunit) + '_' + str(args.rseed)


    if args.saveimage:
        lowbfr_path = pathdir + '/low_traj_img'
        bfr_path = pathdir + '/traj_img'
    elif args.savecmp:
        lowbfr_path = pathdir + '/low_traj_sa'
        bfr_path = pathdir + '/low_traj_sa'
    else:
        lowbfr_path = pathdir + '/low_traj'
        bfr_path = pathdir + '/traj'

    bfrexp = ["vistraj", "visterm", "visvis", "visfval"]
    bfrexp_ = bfrexp + ["train"]
    if args.exp == "generate":
        print('restoring', bfr_path)
        bfr = ExperienceBuffer()
        if args.savecmp:
            bfr.restore_sa(bfr_path)
        else:
            bfr.restore(bfr_path)        
    elif args.exp in bfrexp_:
        if args.exp in bfrexp and args.reverse:
            lowbfr_path = lowbfr_path + 'rev'
        print('restoring', lowbfr_path)
        low_bfr = ExperienceBuffer()
        if args.savecmp:
            low_bfr.restore_sao(lowbfr_path)
        else:
            low_bfr.restore(lowbfr_path)

        mix_traj = False
        if mix_traj:
            low_bfr2 = ExperienceBuffer()
            opdir2 = prefix + '/vis/' + args.task + 'option0' + dirop + str(args.ffuncnunit) + '_' + str(args.rseed)
        #     # TODO: savecmp not supported
        #     low_bfr2.restore(opdir2 + '/low_traj')
    else:
        print('No buffer retrieved')

    #################################
    # 2. Retrieve options
    #################################
    # Experiments which require 1 option to retrieve
    oneopexp = ["visop", "visfval", "train"]
    # Multilpe options to retrieve (But it is retrieved inside the util.py, so let's forget it here)
    # multiopexp = ["sample"]

    if args.exp in oneopexp:
        op = CoveringOption(sess=None, experience_buffer=None, obs_dim=state_dim, obs_bound=mdp.bounds(), num_actions=num_actions, action_dim=action_dim, action_bound=action_bound, low_method=args.lowmethod, f_func=args.ffunction, n_units=args.ffuncnunit, init_all=args.initall, init_around_goal=args.init_around_goal, init_dist=args.init_dist, term_dist=args.term_dist, restore=True, name='option' +  str(args.noptions) + '_' + str(args.ffuncnunit) + '_' + str(args.rseed))
        
        op.restore(opdir)
    else:
        print('No option retrieved')

    #################################
    # 3. Run experiments
    #################################
    
    if args.exp == 'sample':
        print('sample')
        bfr, low_bfr = sample_option_trajectories(mdp, args, noptions=args.noptions)
    elif args.exp == 'generate':
        print('generate_option')
        print('buffersize = ', bfr.size())
        # TODO: option_b_size is the batch size for training f-function. 
        op = CoveringOption(sess=None, experience_buffer=bfr, option_b_size=32, sp_training_steps=args.sptrainingstep, obs_dim=state_dim, obs_bound=state_bound, num_actions=num_actions, action_dim=action_dim, action_bound=action_bound, low_method=args.lowmethod, f_func=args.ffunction, n_units=args.ffuncnunit, init_all=args.initall, reversed_dir=args.reverse, init_around_goal=args.init_around_goal, init_dist=args.init_dist, term_dist=args.term_dist, restore=None, name='option' + str(args.noptions) + '_' + str(args.ffuncnunit) + '_' + str(args.rseed))

    elif args.exp == 'train':
        print('train_option')
        op.reversed_dir = args.reverse
        _, _, r, _, _ = low_bfr.sample(32)
        print('background rewards=', r)
        for _ in range(args.sptrainingstep):
            op.train(low_bfr, batch_size=min(args.batchsize, low_bfr.size()))
    elif args.exp == 'evaloff' or args.exp == 'evalon':
        print('evaloff')
        agent_name = str(args.noptions) + 'options'
        if args.exp == 'evalon':
            agent_name = agent_name + '-online'

        if args.random_agent:
            oagent = GenerateRandomAgent(num_actions, action_dim, action_bound)
        else:
            oagent = OptionAgent(sess=None, obs_dim=state_dim, obs_bound=state_bound, num_actions=num_actions, action_dim=action_dim, action_bound=action_bound, num_options=1 + args.noptions, high_method=args.highmethod, low_method=args.lowmethod, f_func=args.ffunction, batch_size=args.batchsize, buffer_size=args.buffersize, low_update_freq=args.lowupdatefreq, option_batch_size=args.obatchsize, option_buffer_size=args.obuffersize, high_update_freq=args.highupdatefreq, init_all=args.initall, init_around_goal=args.init_around_goal, init_dist=args.init_dist, term_dist=args.term_dist, name=agent_name)
            oagent.reset()

        if args.exp == 'evaloff':
            for nop in range(1, args.noptions + 1):
                op = CoveringOption(sess=None, experience_buffer=None, obs_dim=state_dim, obs_bound=mdp.bounds(), num_actions=num_actions, action_dim=action_dim, action_bound=action_bound, low_method=args.lowmethod, f_func=args.ffunction, init_all=args.initall, init_around_goal=args.init_around_goal, init_dist=args.init_dist, term_dist=args.term_dist, restore=True, name='option' + str(nop) + '_'+ str(args.ffuncnunit) + '_' + str(args.rseed))

                if args.reverse:
                    opdir = prefix + '/vis/' + args.task + 'option' + str(nop) + 'rev_' + str(args.ffuncnunit) + '_' + str(args.rseed)
                else:
                    opdir = prefix + '/vis/' + args.task + 'option' + str(nop) + '_' + str(args.ffuncnunit) + '_' + str(args.rseed)                

                op.restore(opdir)
                print('restored option', opdir)
                oagent.add_option(op)
        else:
            print('evalon')
        mdp.reset()
        run_agents_on_mdp([oagent], mdp, episodes=args.nepisodes, steps=args.nsteps, instances=args.ninstances, cumulative_plot=True, verbose=args.verbose)
    else:
        print('No experiments run')

    #################################
    # 4. Plot figures
    #################################
    if args.exp == 'visop':
        plot_op(op, args, mdp, state_bound, opdir + '/eigenfunc.pdf')
    elif args.exp == 'vistraj' or args.exp == 'visterm':
        print(args.exp)
        samples = low_bfr.buffer
        size = low_bfr.size()
        trajectories = []
        cur_o = None
        for i in range(size):
            s, _, _, _, t, o = samples[i][0], samples[i][1], samples[i][2], samples[i][3], samples[i][4], samples[i][5]
            if cur_o == args.noptions:
                if o == args.noptions and not t and i != size - 1:
                    traj.append(s)
                else:
                    # traj.append(s2)
                    # if args.tasktype == 'pinball':
                    #     t = [s for s in traj if s.x != 0.2 or s.y != 0.2] # TODO: hack to remove the init state.
                    # else:
                    #     t = traj
                    if len(traj) > 10:
                        trajectories.append((i, traj))
                    
                    cur_o = 0
                    traj = []
            else:
                if o == args.noptions:
                    traj = [s]
                    cur_o = args.noptions

        if len(trajectories) == 0:
            print('no trajectories sampled')

        if args.exp == 'visterm':
            terms = [traj[-1] for traj in trajectories]
            terms = terms[0:min(len(terms), 100)]
            # print('terms=', type(terms))
            print('#terms=', len(terms))
            if args.reverse:
                plot_terms(terms, mdp, args, filename=pathdir + '/' + 'terms' + 'rev')
            else:
                plot_terms(terms, mdp, args, filename=pathdir + '/' + 'terms')
        else:
            t = trajectories[1][1]
            plot_traj(t, mdp, args, filename=pathdir + '/' + 'traj' + str(1))

    elif args.exp == 'visvis':
        print('visvis')
        samples = low_bfr.buffer
        traj = [samples[i][0] for i in range(low_bfr.size())]
        if mix_traj:
            
            samples2 = low_bfr2.buffer
            traj2 = [samples2[i][0] for i in range(int(min(low_bfr2.size() / 2, len(traj) / 2)))]

            traj = traj[:int(len(traj) / 2)] + traj2
        plot_vis(traj, args, mdp, pathdir + '/visitation')
    elif args.exp == 'visfval':
        print('visfval')
    else:
        print('No plots')

    #################################
    # 5. Save the results
    #################################
    if args.exp == 'sample':
        print('save sample')
        if args.reverse:
            dirop = "rev"
        else:
            dirop = ""
        
        if args.saveimage:
            bfr.save(pathdir + '/traj_img' + dirop)
            low_bfr.save(pathdir + '/low_traj_img' + dirop)
        elif args.savecmp:
            bfr.save_sa(pathdir + '/traj_sa' + dirop)
            low_bfr.save_sao(pathdir + '/low_traj_sa' + dirop)
        else:
            bfr.save(pathdir + '/traj' + dirop)
            low_bfr.save(pathdir + '/low_traj' + dirop)
            
    elif args.exp == 'evaloff' or args.exp == 'evalon':
        print('save',  args.exp)
        options = oagent.options
        for nop in range(1, len(options)):
            opdir = prefix + '/vis/' + args.task + 'option' + str(nop) + '_' + str(args.ffuncnunit) + '_' + str(args.rseed)
            if args.exp == 'evalon':
                opdir = opdir + '_online'

            options[nop].save(opdir + '_trained')
        oagent.option_buffer.save(pathdir + '_trained' + '/' + 'traj')
        oagent.experience_buffer.save(pathdir + '_trained' + '/' + 'low_traj')        
    elif args.exp == 'generate':
        print('save generate')
        op.save(opdir)
    elif args.exp == 'train':
        print('save train')
        if args.reverse:
            op.save(opdir, rev=True)
        else:
            op.save(opdir)
    else:
        print('No save')
        

if __name__ == "__main__":
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.ERROR)
    main(open_plot=not sys.argv[-1] == "no_plot")
