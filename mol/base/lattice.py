import numpy as np
from numpy import outer, dot, cross, arctan2
from numpy.linalg import norm
from mol.base.linear import angle, normalized

class Lattice(object):
    def __init__(self, lattice):
        self.lattice = lattice
        self.reciprocal_lattice = reciprocal(lattice)
        self.volume = volume(lattice)
        self.reciprocal_volume = volume(self.reciprocal_lattice)


def direct_to_cartesian(coordinates, lattice):
    return coordinates @ lattice


def cartesian_to_direct(coordinates, reciprocal_lattice):
    return coordinates @ reciprocal_lattice.T / np.pi / 2


def volume(lattice):
    return dot(cross(lattice[0], lattice[1]), lattice[2])


def reciprocal(lattice):
    return 2 * np.pi * np.array([cross(lattice[1], lattice[2]),
                                 cross(lattice[2], lattice[0]),
                                 cross(lattice[0], lattice[1])]) / volume(lattice)

def lattice_constant(lattice):
    abc = np.array([norm(l) for l in lattice])
    abg = np.array([angle(lattice[1], lattice[2]),
                    angle(lattice[2], lattice[0]),
                    angle(lattice[0], lattice[1])])
    return abc, abg


def fractional_to_cartesian(lattice, vector):
    return vector.dot(lattice)


def cartesian_to_fractional(reciprocal_lattice, vector):
    return vector.dot(reciprocal_lattice.T) / (2 * np.pi)


def test():
    # l = np.array([[2.0, 0.1, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
    lat = np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]]) * 3.7    
    rec = reciprocal(lat)
    a = np.array([[0.5, 0.0, 0.5],
                  [0.0, 0.5, 0.5]])
    a = np.array([0.5, 0.0, 0.5])
    aa = fractional_to_cartesian(lat, a)
    print(a)
    print(aa)
    print(cartesian_to_fractional(rec, aa))



if __name__ == '__main__':
    test()
