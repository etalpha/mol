import numpy as np
from numpy import outer, dot
from numpy.linalg import norm

def sr1(fg_eval, x0, conv=1e-4):
    x = x0
    h = np.eye(len(x0)) * 0.1
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
        b = dx - h.dot(y)
        h = h + outer(b, b) / b.dot(y)
        if max(abs(dx)) < conv:
            return x, f, g, h
        print(f"{j}: {x}")
        j += 1
