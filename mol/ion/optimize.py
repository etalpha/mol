import numpy as np
from numpy.linalg import solve


def steepest_descent_step(g, alpha):
    return - alpha * g

def newton_step(g, h):
    return - solve(h, gs[-1])
