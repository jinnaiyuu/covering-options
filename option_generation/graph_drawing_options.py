import numpy as np
from scipy.optimize import minimize
import math

from options.graph.graph_drawing_objective import GraphDrawingObjective
from options.util import GetRandomWalk

def GraphDrawingOptions(A, k=1):
    delta = 0.05
    beta = 5.0
    N = A.shape[0]
    dim = int(math.floor(k / 2))

    rho = np.full(N, 1.0/N, dtype=float)
    P = GetRandomWalk(A)
    GDO = GraphDrawingObjective(rho, P, delta, beta)

    x0 = np.full(int(N) * dim, 0.1)
    res = minimize(GDO, x0)
    # res = minimize(GDO, x0, method='Nelder-Mead', options={'maxiter': 50}, tol=1e-5)

    sol = res.x.reshape((dim, N))

    print('sol=', sol)
    B = A.copy()
    options = []
    for i in range(dim):
        vmin = np.argmin(sol[i])
        vmax = np.argmax(sol[i])
        options.append((vmin, vmax))
        B[vmin][vmax] = 1
        B[vmax][vmin] = 1
    
    return B, options


if __name__ == '__main__':
    A = np.zeros((9, 9), dtype=float)
    
    A[0, 1] = 1.0
    A[0, 3] = 1.0    
    A[1, 0] = 1.0
    A[1, 2] = 1.0
    A[1, 4] = 1.0
    A[2, 1] = 1.0
    A[2, 5] = 1.0
    A[3, 0] = 1.0
    A[3, 4] = 1.0
    A[3, 6] = 1.0
    A[4, 1] = 1.0
    A[4, 3] = 1.0
    A[4, 5] = 1.0
    A[4, 7] = 1.0
    A[5, 2] = 1.0
    A[5, 4] = 1.0
    A[5, 8] = 1.0
    A[6, 3] = 1.0
    A[6, 7] = 1.0
    A[7, 4] = 1.0
    A[7, 6] = 1.0
    A[7, 8] = 1.0
    A[8, 5] = 1.0
    A[8, 7] = 1.0


    B, options = GraphDrawingOptions(A, k=4)
    print('A=', A)
    print('B=', B)
    print('options=', options)


    
