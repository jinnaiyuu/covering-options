import numpy as np
from options.util import neighbor

class MatrixMDP():
    def __init__(self, matrix):
        self.matrix = matrix
        self.N = int(matrix.shape[0])

    def available_actions(self, s):
        n = neighbor(self.matrix, s)
        return n

    def initial_state(self):
        return 0
    
    def next_state(self, s, a):
        if self.is_goal(a):
            return a, 0
        else:
            return a, -1

    def is_goal(self, s):
        return s == self.N - 1

    
