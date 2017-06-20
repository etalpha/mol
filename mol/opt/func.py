import sympy as sym
import numpy as np

class AutoDiff(object):
    def __init__(self, x, f):
        self.define(x, f)

    def define(self, x, f):
        self.x = x
        self._f = f
        self._g = [sym.diff(self._f, x) for x in self.x]
        self._h = [[sym.diff(sym.diff(self._f, x1), x2) for x2 in self.x] for x1 in self.x]

    def f(self, x):
        return self._f.subs(zip(self.x, x))

    def g(self, x):
        return np.array([g.subs(zip(self.x, x)) for g in self._g], dtype=np.float64)

    def h(self, x):
        return np.array([[h.subs(zip(self.x, x)) for h in hs] for hs in self._h], dtype=np.float64)
    
    def fg(self, x):
        return self.f(x), self.g(x)

    def fgh(self, x):
        return self.f(x), self.g(x), self.h(x)

class ExpDiphasic(AutoDiff):
    def __init__(self, alpha):
        xs = sym.MatrixSymbol('x', 2, 1)
        x = xs[0]
        y = xs[1]
        f = - sym.exp(-(x-1)**2-y**2) - sym.exp(-(x+1)**2-y**2) + 0.001 *((x-1)**2 + (x+1)**2 + 2*y**2)
        super().__init__(xs, f)

