import numpy as np
from collections import OrderedDict
from mol.base.atom import Atom, Atoms, atomic_number

class Poscar(object):
    def __init__(self, comment, lattice, atoms, selective_dynamics_flags=None):
        self.comment = comment
        self.lattice = lattice
        self.atoms = atoms
        self.selective_dynamics_flags = selective_dynamics_flags

def read_comment(f):
    return f.readline().strip()


def read_universal_scaling_factor(f):
    return float(f.readline().strip())


def read_lattice_line(f):
    return np.array([float(s) for s in f.readline().split()[0:3]], dtype=np.float64)


def read_lattice(f):
    return np.array([read_lattice_line(f) for _ in range(3)])


def read_element(f):
    element_names = f.readline().split()
    element_numbers = [int(s) for s in f.readline().split()]
    assert len(element_names) == len(element_numbers)
    return OrderedDict(zip(element_names, element_numbers))


def read_selective_cartesian(f):
    l1 = f.readline().strip()
    if l1[0] in ("s", "S"):
        selective_dynamics = True
        l2 = f.readline().strip()
        cartesian = True if l2[0] in ("C", "c", "K", "k") else False
    else:
        selective_dynamics = False
        cartesian = True if l1[0] in ("C", "c", "K", "k") else False
    return selective_dynamics, cartesian


def read_coordinate(f):
    return np.array([float(s) for s in f.readline().split()])


def read_true_false(s):
    if s == 'T':
        return True
    elif s == 'F':
        return False
    else:
        raise ValueError(f"T or F is expected. arg is {s}")


def read_coordinate_flag(f):
    coordinate_flag = f.readline().split()
    coordinate = np.array([float(s) for s in coordinate_flag[0:3]])
    flag = np.array([read_true_false(s) for s in coordinate_flag[3:6]])
    return coordinate, flag


def read_coordinates(f, n):
    return [read_coordinate(f) for _ in range(n)]


def read_coordinates_flags(f, n):
    coordinates = []
    flags = []
    for _ in range(n):
        coordinate, flag = read_coordinate_flag(f)
        coordinates.append(coordinate)
        flags.append(flag)
    return np.array(coordinates), np.array(flags)


def vectors_transform(v, a):
    return v @ a


def atomic_numbers(element):
    numbers = []
    for elem, num in element.items():
        n = atomic_number(elem)
        for i in range(num):
            numbers.append(n)
    return np.array(numbers, dtype=np.int64)


def read_poscar(path):
    with open(path) as f:
        comment = read_comment(f)
        universal_scaling_factor = read_universal_scaling_factor(f)
        lattice = read_lattice(f)
        element = read_element(f) # Orderd Dict of name and num
        selective_dynamics, cartesian = read_selective_cartesian(f)
        n_atoms = sum(element.values())
        if selective_dynamics:
            coordinate, flag = read_coordinates_flags(f, n_atoms)
        else:
            coordinate = read_coordinates(f, n_atoms)
            flag = None
        if not cartesian:
            coordinate = vectors_transform(coordinate, lattice)
        coordinate *= universal_scaling_factor
        lattice *= universal_scaling_factor
        numbers = atomic_numbers(element)
        # atoms = [Atom(n, r) for n, r in zip(numbers, coordinate)]
        atoms = Atoms(numbers, coordinate)
        return Poscar(comment, lattice, atoms, flag)

def write_poscar(path):
    ...
