#!/usr/bin/env python

# Python imports.
import sys
import time
from collections import OrderedDict
import copy

# Libraries
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Other imports.
from simple_rl.tasks import GridWorldMDP
from simple_rl.tasks.grid_world.GridWorldMDPClass import make_grid_world_from_file
from simple_rl.planning.ValueIterationClass import ValueIteration

from options.graph.steiner_tree import DiameterConstrainedSteinerTree, MinimumWeightMatching
from options.option_generation.util import GetAdjacencyMatrix
from options.util import GetCost

def dump_options(filename, C, intToS):
    with open(filename, 'w') as f:
        for s in range(C.shape[0]):
            if C[s] == 1:
                state = intToS[s]
                print(state)
                w = str(state.x) + ' ' + str(state.y) + '\n'
                f.write(w)

def StatesToArray(intToS, states):
    N = len(intToS)
    arr = np.zeros(N, dtype=int)

    print('intToS', intToS)
    print('states', states)
    for s in states:
        for i, d in enumerate(intToS):
            if d.x == s[0] and d.y == s[1]:                
                arr[i] = 1
                print('d=', d, ', i=', i)
                continue
    return arr


def TestMatching():
    domain = '5x5grid'
        
    fname = '../tasks/' + domain + '.txt'    
    mdp = make_grid_world_from_file(fname)

    G, intToS = GetAdjacencyMatrix(mdp)
    c = GetCost(G)


    matrix, F, LB = MinimumWeightMatching(G, c)

    print('F\'=', F)
    print('LB=', LB)

    Gnx = nx.from_edgelist(F)
    dic = dict()
    for i, s in enumerate(intToS):
        dic[i] = (s.x, s.y)

    nx.draw_networkx_nodes(Gnx, pos=dic, node_size=300, node_color='g')
    nx.draw_networkx_edges(Gnx, pos=dic)
    
    plt.savefig('Matching.pdf')
    

if __name__ == "__main__":
    TestMatching()
    exit(0)
    # domain = '5x5grid'
    # goals = [(1, 5), (1, 1), (5, 5), (3, 3), (5, 1)]
    
    domain = '9x9grid'
    goals = [(1, 1), (1, 9), (9, 1), (9, 9), (5, 5)]
    
    # domain = 'fourroom'
    # goals = [(1, 1), (1, 11), (11, 1), (11, 11), (5, 5), (8, 7), (5, 7)]
    
    fname = '../../tasks/' + domain + '.txt'    
    mdp = make_grid_world_from_file(fname)

    G, intToS = GetAdjacencyMatrix(mdp)
        
    c = np.ones_like(G, dtype=int)
    d = GetCost(G)
    # print('d=', d)
    # TODO
    K = StatesToArray(intToS, goals)
    # K = np.random.binomial(n=1, p=0.2, size=G.shape[0]) # np.ones(G.shape[0], dtype=int)

    print('K=', K)
    D = 15

    tree, options = DiameterConstrainedSteinerTree(G, c, d, K, D, 0.1)

    print('tree', tree)


    # #######################
    # # Visualize generated options
    # xys = []
    # # TODO: Convert the state id into
    # for o in options:
    #     init = o[0]
    #     term = o[1]
    #     print('o = ', intToS[init], intToS[term])
    #     x = intToS[init].x
    #     y = intToS[init].y
    #     xys.append((y, x))
    #     x = intToS[term].x
    #     y = intToS[term].y
    #     xys.append((y, x))
    # 
    # # TODO: Output files in pdf format
    # 
    # # TODO: Write a code to show the options in line instead of points
    # mdp.visualize(xys, domain + '-' + 'SteinerTree')

    ######################
    # Show a tree?

    tree[tree == -1] = 0

    nxTree = nx.to_networkx_graph(tree)

    dic = dict()
    for i, s in enumerate(intToS):
        dic[i] = (s.x, s.y)

    goal_list = np.argwhere(K == 1).flatten().tolist()
    
    print('goals=', goal_list)
    # nx.draw(nxTree, pos=dic)
    nx.draw_networkx_nodes(nxTree, pos=dic, node_size=100, node_color='g')
    nx.draw_networkx_nodes(nxTree, pos=dic, nodelist=goal_list, node_size=300, node_color='r')
    nx.draw_networkx_edges(nxTree, pos=dic)
    
    plt.savefig('connectivity.pdf')
