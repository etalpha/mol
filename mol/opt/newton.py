import numpy as np
from numpy.linalg import solve, norm
from func import ExpDiphasic

def newton(fgh_eval, x0, conv=1e-4):
    x = x0
    i = 0
    while True:
        f, g, h = fgh_eval(x)
        dx = - solve(h, g)
        if norm(dx) < conv:
            return x, f, g, h
        x = x + dx
        print(f"{i}: {x}")
        i += 1

def test():
    fgh = ExpDiphasic(1)
    x0 = [0.5, 0.4]
    newton(fgh.fgh, x0)

if __name__ == '__main__':
    test()
