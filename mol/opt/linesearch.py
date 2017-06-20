import numpy as np
from numpy.linalg import solve
import sympy as sym

def wolfe_armijo(c1, alpha, f0, f1, p0, g0):
    return f1 <= f0 + c1 * alpha * p0.dot(g0)

def wolfe_curvature(c2, alpha, g0, g1, p0):
    return - p0.dot(g1) <= - c2 * p0.dot(g0)

def strong_wolfe_curvature(c2, g0, g1, p0):
    return abs(p0.dot(g1)) <= c2 * abs(p0.dot(g0))

def diff0(x, n):
    return [x ** i for i in range(n)]

def diff1(x, n):
    return [i * x ** (i - 1) if i > 0 else 0 for i in range(n)]

def coefficient_matrix(xs):
    n = len(xs) * 2
    return np.array([diff0(x, n) for x in xs] + [diff1(x, n) for x in xs], dtype=np.float64)

def coefficients(xs, ys, yds):
    n = len(xs)
    assert len(ys) == n and len(yds) == n
    return solve(coefficient_matrix(xs), np.array([ys, yds], dtype=np.float64).flatten())

def poly_fit(xs, ys, yds, x):
    cs = coefficients(xs, ys, yds)
    return sum(c * x ** i for i, c in enumerate(cs))

def extremes_prediction(xs, fs, gs):
    a = sym.Symbol('a')
    poly0 = poly_fit(xs, fs, gs, a)
    poly1 = sym.diff(poly0, a)
    poly2 = sym.diff(poly1, a)
    es = np.array([
        ss.real for ss in [
        complex(s) for s in sym.solve(poly1, a)]
        if ss.imag == 0.0], dtype=np.float64)
    s0 = np.array([poly0.subs([(a, e)]) for e in es])
    s1 = np.array([poly1.subs([(a, e)]) for e in es])
    s2 = np.array([poly2.subs([(a, e)]) for e in es])
    return es, s0, s1, s2

def minimums_prediction(xs, fs, gs):
    es, d0s, d1s, d2s = extremes_prediction(xs, fs, gs)
    cond = d2s > 0.0
    return es[cond], d0s[cond], d1s[cond], d2s[cond]

def forward_minimums_prediction(xs, fs, gs):
    es, d0s, d1s, d2s = minimums_prediction(xs, fs, gs)
    cond = es > 0.0
    return es[cond], d0s[cond]

def predict_min(xs, fs, gs):
    es, s0 = forward_minimums_prediction(xs, fs, gs)
    if len(es) == 0:
        return None
    else:
        i = s0.argmin()
        return es[i]

def line_search(fg_eval, alpha, f0, g0, x0, p, c1=0.0001, c2=0.9):
    xs = [x0]
    fs = [f0]
    gs = [g0]
    alphas = [0.0]
    psis = [f0]
    psids = [g0.dot(p)]
    while True:
        xs.append(x0 + alpha * p)
        f, g = fg_eval(xs[-1])
        fs.append(f)
        gs.append(g)
        alphas.append(alpha)
        psis.append(f)
        psids.append(g.dot(p))
        alpha = predict_min(alphas, psis, psids)
        if alpha is None:
            return xs, fs, gs
        elif wolfe_armijo(c1, alpha, f0, fs[-1], p, g0):
            return xs, fs, gs
        elif wolfe_curvature(c2, alpha, g0, gs[-1], p):
            return xs, fs, gs
        else:
            pass
            


def test():
    xs = [0, 1]
    ys = [1., -0.5]
    yds = [-1., -1.]
    m = predict_min(xs, ys, yds)
    print(m)


if __name__ == '__main__':
    test()
