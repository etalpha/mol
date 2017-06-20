import sympy as sym
import numpy as np
import numpy.linalg as nl


def vector(name, length):
    return np.array([f"{name}_{i}" for i in range(length)])

def norm2(r):
    return sum(r ** 2)

def norm(r):
    return sym.sqrt(norm2(r))

def normalize(r):
    return r / norm(r)

def cartesian(rtp):
    r, theta, phi = rtp
    cos = sym.cos
    sin = sym.sin
    return np.array([r * sin(theta) * cos(phi),
                     r * sin(theta) * sin(phi),
                     r * cos(theta)])

def substitute(s_out, s_inp, n_inp):
    sub = {s: n for s, n in zip(s_inp, n_inp)}
    return np.array([s.subs(sub) for s in s_out.flatten()], dtype=np.float64).reshape(s_out.shape)

def jacobi(out, inp):
    return np.array([[sym.diff(o, i) for i in inp] for o in out])

def find_root(out, s_in, n_in, jacobian, conv):
        while True:
            n_jacobian = substitute(jacobian, s_in, n_in)
            f = substitute(out, s_in, n_in)
            if np.max(np.abs(f)) < conv:
                return n_in
            din = nl.solve(n_jacobian, f)
            n_in = n_in - din

def diff_backward(grad_out, jacobian, s_in, n_in):
    return substitute(jacobian, s_in, n_in).T @ grad_out
