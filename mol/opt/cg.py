import numpy as np
from numpy import dot
from func import ExpDiphasic
import linesearch as ls


def cg(fg_eval, x0, conv=1e-4):
    x = x0
    f, g = fg_eval(x0)
    p = -g
    k = 0
    while True:
        xs, fs, gs = ls.line_search(fg_eval, 0.5, f, g, x, p, c1=0.0001, c2=0.9)
        # alpha = 0.1
        # x = x + alpha * p
        x = xs[-1]
        g_old = g
        f, g = fg_eval(x)
        beta = dot(g, g) / dot(g_old, g_old)
        p = -g + beta * p
        k = k + 1
        print(f"{k}: {x}")
        if max(abs(p)) < conv:
            return x, f, g
        
def test():
    fg = ExpDiphasic(1)
    x = [0.1, 0.1]
    cg(fg.fg, x)

if __name__ == '__main__':
    test()
