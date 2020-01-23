
import random
import numpy as np
import networkx as nx
from options.graph.spectrum import ComputeFiedlerVector, ComputeConnectivity
from options.util import AddEdge, neighbor

def ComputeCoverTimeS(G, s, sample=1000):
    ##########################
    # PLEASE WRITE A TEST CODE
    ##########################
    '''
    Args:
        G (numpy 2d array): Adjacency matrix (may be an incidence matrix).
        s (integer): index of the initial state
        sample (integer): number of trajectories to sample
    Returns:
        (float): the expected cover time from state s
    Summary:
        Given a graph adjacency matrix, return the expected cover time starting from node s. We sample a set of trajectories to get it.
    '''
    
    N = G.shape[0]

    n_steps = []
    
    for i in range(sample):
        visited = np.zeros(N, dtype=int)
        cur_s = s
        cur_steps = 0

        while any(visited == 0):
            s_neighbor = neighbor(G, cur_s)
            next_s = random.choice(s_neighbor)
            visited[next_s] = 1
            cur_s = next_s
            cur_steps += 1
            
        n_steps.append(cur_steps)

    # print('n_steps=', n_steps)

    avg_steps = sum(n_steps) / sample
    return avg_steps

G1 = np.array([[0, 1], [1, 0]])
s1 = 0
s2 = 1

G2 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
s3 = 0
s4 = 1
s5 = 2

def ComputeCoverHelperTest():
    
    if 0.9 < ComputeCoverTimeS(G1, s1, sample=1000) < 1.1:
        print("test for G1, s1, s = 1000 passed ")
    else:
        print("test for G1, s1, s = 1000 failed")

    if 0.9 < ComputeCoverTimeS(G1, s2, sample=1000) < 1.1:
        print("test for G1, s2, s = 1000 passed ")
    else:
        print("test for G1, s2, s = 1000 failed")

    if 1.85 < ComputeCoverTimeS(G2, s3, sample=1000) < 2.15:
        print("test for G2, s3, s = 1000 passed ")
    else:
        print("test for G2, s3, s = 1000 failed")

    if 2.85 < ComputeCoverTimeS(G2, s4, sample=1000) < 3.15:
        print("test for G2, s4, s = 1000 passed ")
    else:
        print("test for G2, s4, s = 1000 failed")
    

    if 1.85 < ComputeCoverTimeS(G2, s5, sample=1000) < 2.15:
        print("test for G2, s3, s = 1000 passed ")
    else:
        print("test for G2, s3, s = 1000 failed")

    




def ComputeCoverTime(G, samples=1000):
    ##########################
    # PLEASE WRITE A TEST CODE
    ##########################
    '''
    Args:
        G (numpy 2d array): Adjacency matrix (or incidence matrix)
    Returns:
        (float): the expected cover time
    Summary:
        Given a graph adjacency matrix, return the expected cover time.
    '''
    N = G.shape[0]

    c_sum = 0

    for i in range(samples):
        init = random.randint(0, N-1)
        c_i = ComputeCoverTimeS(G, init, sample=1)
        c_sum += c_i
        
    return float(c_sum) / float(samples)

if __name__ == "__main__":

    # PlotConnectivityAndCoverTime(100)
    # exit(0)
    
    Gnx = nx.path_graph(4)
    
    graph_ = nx.to_numpy_matrix(Gnx)
    graph = np.asarray(graph_)

    v = ComputeFiedlerVector(Gnx) # numpy array of floats
    
    augGraph = AddEdge(graph, np.argmax(v), np.argmin(v))
    

    # print('Graphs')
    # print(graph)
    # print(augGraph)
    t2 = ComputeCoverTime(augGraph)
    print('CoverTime Aug1', t2)
    lb2 = nx.algebraic_connectivity(nx.to_networkx_graph(augGraph))
    print('lambda        ', lb2)
    ComputeCoverHelperTest()
