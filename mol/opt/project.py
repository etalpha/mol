import numpy as np
from numpy import dot, outer, cross
from numpy.linalg import solve, norm


def diff_matrix(mat):
    return np.array([mat[i + 1] - mat[i] for i in range(len(mat) - 1)])

def project_quasi_newton(fg_eval, x0, alpha, conv=1e-4):
    xs = [x0]
    f0, g0 = fg_eval(x0)
    fs = [f0]
    gs = [g0]
    xs.append(x0 - alpha * gs[-1])
    j = 0
    while True:
        f, g = fg_eval(xs[-1])
        fs.append(f)
        gs.append(g)
        DX = diff_matrix(xs)
        DG = diff_matrix(gs)
        dx = -DX.T.dot(solve(DG.dot(DG.T), DG)).dot(g)
        xs.append(xs[-1] + dx)
        print(f"{j}: {xs[-1]}")
        if max(abs(dx)) < conv:
            return xs, fs, gs
        j += 1

