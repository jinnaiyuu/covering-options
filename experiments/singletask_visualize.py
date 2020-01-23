# Python imports.
import sys
import time
from collections import OrderedDict
import copy

# Libraries
import numpy as np

# Other imports.
from simple_rl.tasks import GridWorldMDP
from simple_rl.tasks.grid_world.GridWorldMDPClass import make_grid_world_from_file

# Options
from options.option_generation.vi_distance import get_distance
from options.util import GetRadius

def calculate_iters(mdp, options):
    sToInt, intToS, distance = get_distance(mdp)

    n = len(sToInt)
    C = np.zeros(n, dtype=int)

    for op in options:
        for i in range(n):
            state = intToS[i]
            if state.x == op[0] and state.y == op[1]:
                C[i] = 1
                continue
    
    R = GetRadius(distance.transpose(), C)
    return R
    

if __name__ == "__main__":

    # domain = 'Imaze' # OP/VI = [1, 2, 3, 4, 5], [6, 4, 3, 3, 2]
    domain = 'fourroom' # OP/VI = [1, 2, 3, 4, 5], [17, 11, 10, 8, ?]
    # domain = 'fourroom_s'
    # domain = '9x9grid' # OP/VI = [1, 2, 3, 4, 5], [11, 9, 8, 7, 6]
    # domain = '5x5grid'
    # domain = '3x3grid'
    
    fname = '../tasks/' + domain + '.txt'    
    mdp = make_grid_world_from_file(fname)

    # mdp = TaxiOOMDP(5, 5, agent={"x":1, "y":1, "has_passenger":0}, walls=[], passengers=[{"x":4, "y":2, "dest_x":1, "dest_y":4, "in_taxi":0}, {"x":1, "y":5, "dest_x":4, "dest_y":3, "in_taxi":0}])
    
    print('MDP =', mdp)

    iters = 13
    # solver = 'eigen'
    # solver = 'bet'
    # solver = 'optimal'
    solver = 'chvatal'


    ops = 4
    # solver = 'archer'
    
    options = []

    # filename = '../data/' + domain + '_' + str(13) + solver
    # filename = 'data/' + domain + '_' + str(ops) + 'ops_' + solver
    filename = '../data/' + domain + '_' + str(iters) + 'iter_' + solver
    with open(filename) as f:
        for line in f:            
            x, y = map(int, line.split())
            options.append((x, y))

    options = options[0:ops]
            
    R = calculate_iters(mdp, options)
    print('#iters =', R)
    
    # TODO: Read from files
    mdp.visualize(options, filename)
