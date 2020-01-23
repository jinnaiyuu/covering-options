import numpy as np
import itertools
from scipy.optimize import minimize
from options.util import GetRandomWalk

def Attr(rho, P, fs):
    # rho: numpy array of size N. prob.
    # P  : numpy array of size NxN. each row being a prob.
    # F  : list of numpy arrays of size N (TODO: should this be a numpy array?)
    ret = 0.0
    N = rho.shape[0]

    for u in range(N):
        for v in range(N):
            prob = rho[u] * P[u, v]
            # ret += (F[u] - F[v]) * (F[u] - F[v]) # TODO: 1 dimensional for now
            for f in fs:
                ret += (f[u] - f[v]) * (f[u] - f[v])
            
    return ret / 2.0

def Repl(rho, P, delta, fs):
    ret = 0.0
    N = rho.shape[0]
    
    for u in range(N):
        for v in range(N):
            prob = rho[u] * rho[v] # For repulsive term, we take exp. over rhos.
            for j in range(len(fs)):
                for k in range(j, len(fs)):
                    f1 = fs[j]
                    f2 = fs[k]
                    if j == k:
                        res = delta
                    else:
                        res = 0
                    ret += (f1[u] * f2[u] - res) * (f1[v] * f2[v] - res)
    return ret

def GraphDrawingObjective(rho, P, delta, beta):
    # TODO: delta should be a function instead of a constant value
    N = rho.shape[0]
    def GDO(F):
        fs = []
        for k in range(int(F.shape[0] / N)):
            f = F[N * k:N * (k+1)]
            fs.append(f)
        return Attr(rho, P, fs) + beta * Repl(rho, P, delta, fs)
    return GDO

if __name__ == "__main__":
#     rho = np.array([0.25, 0.50, 0.25])    
#     P = np.array([[0.0, 1.0, 0.0],
#                   [0.5, 0.0, 0.5],
#                   [0.0, 1.0, 0.0]])

    rho = np.full(9, 1.0/9.0, dtype=float)
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

    P = GetRandomWalk(A)

    print('P=', P)

    delta = 0.1
    beta = 5.0

    GDO_fn = GraphDrawingObjective(rho, P, delta, beta)

    dim = 3

    x0 = np.full(int(rho.shape[0]) * dim, 0.1)
    res = minimize(GDO_fn, x0, method='nelder-mead')


    sol = res.x.reshape((dim, int(rho.shape[0])))
    print('solution=\n', sol)
    # gdo_val = GDO_fn([f1, f2])
    # print('gdo=', gdo_val)


    # For our purpose, we want to draw an edge from minimum to maximum.
