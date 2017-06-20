import numpy as np
from numpy import outer, dot
from numpy.linalg import norm

def bfgs(fg_eval, x0, conv=1e-6, alpha=0.01):
    x = x0
    h = np.eye(len(x0)) * alpha
    i = np.eye(len(x0))
    j = 0
    f, g = fg_eval(x0)
    g_old = g
    while True:
        dx = - h.dot(g)
        x = x + dx
        f, g = fg_eval(x)
        y = g - g_old
        g_old = g
        numerator = dot(y, dx)
        if numerator > 0.0:
            b = i - (outer(y, dx) / dot(y, dx))
            h = b.T.dot(h).dot(b) + (outer(dx, dx) / dot(y, dx))
        else:
            b = i - i/2
            h = b.T.dot(h).dot(b) + i/2
        if max(abs(dx)) < conv:
            return x, f, g, h
        print(f"{j}: {x}")
        j += 1
