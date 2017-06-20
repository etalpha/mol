import numpy as np


class Atom(object):
    def __init__(self, n, x, g=np.zeros(3, dtype=np.float64)):
        self.n = np.asarray(n, dtype=np.int64)
        self.x = np.asarray(x, dtype=np.float64)
        self.g = np.asarray(g, dtype=np.float64)

    @property
    def element(self):
        return element_names[self.n - 1]

    @property
    def weight(self):
        return element_weights[self.n - 1]

    def __repr__(self):
        return f"{self.element}{self.x}"

class Atoms(object):
    def __init__(self, n, x, g=None):
        self.n = np.asarray(n, dtype=np.int64)
        self.x = np.asarray(x, dtype=np.float64)
        if g is None:
            self.g = np.zeros(self.x.shape, dtype=np.float64)
        else:
            self.g = np.asarray(g, dtype=np.float64)

    def __repr__(self):
        def round(x):
            return ' '.join(['{: 10f}'.format(i) for i in x])
        return '\n'.join([f"{atomic_name(n):>2} {round(x)}" for n, x in zip(self.n, self.x)])

def atomic_number(name):
    return element_names.index(name) + 1

def atomic_name(number):
    return element_names[number - 1]

def atomic_weight(number):
    assert isinstance(number, int)
    return element_weights[number - 1]

element_names = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra']

element_weights = np.array([1.00794, 4.0026, 6.941, 9.01218, 10.811, 12.011, 14.0067, 15.9994, 18.9984, 20.1797, 22.98977, 24.305, 26.98154, 28.0855, 30.97376, 32.066, 35.4527, 39.948, 39.0983, 40.078, 44.9559, 47.88, 50.9415, 51.996, 54.938, 55.847, 58.9332, 58.6934, 63.546, 65.39, 69.723, 72.61, 74.9216, 78.96, 79.904, 83.8, 85.4678, 87.62, 88.9059, 91.224, 92.9064, 95.94, None, 101.07, 102.9055, 106.42, 107.868, 112.41, 114.82, 118.71, 121.757, 127.6, 126.9045, 131.29, 132.9054, 137.33, 138.9055, 140.12, 140.9077, 144.24, None, 150.36, 151.965, 157.25, 158.9253, 162.5, 164.9303, 167.26, 168.9342, 173.04, 174.967, 178.49, 180.9479, 183.85, 186.207, 190.2, 192.22, 195.08, 196.9665, 200.59, 204.383, 207.2, 208.9804, None, None, None, None, 226.0254], dtype=np.float64)

if __name__ == '__main__':
    a = Atom(1, [0, 0, 0])
    print(a.weight)
    print(atomic_number('H'))
