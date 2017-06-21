import numpy as np
from collections import OrderedDict
from mol.base.atom import Atom, Atoms, atomic_number, atoms_to_list, list_to_atoms
import mol.base.atom as atom
from mol.base.lattice import Lattice, direct_to_cartesian, cartesian_to_direct

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
            # coordinate = vectors_transform(coordinate, lattice)
            coordinate = direct_to_cartesian(coordinate, lattice)
        coordinate *= universal_scaling_factor
        lattice *= universal_scaling_factor
        numbers = atomic_numbers(element)
        # atoms = [Atom(n, r) for n, r in zip(numbers, coordinate)]
        atoms = Atoms(numbers, coordinate)
        return Poscar(comment, Lattice(lattice), atoms, flag)


def sorted_atoms(atoms, sort):
    al = atoms_to_list(atoms)
    sal = sorted(al, key=lambda a: sort(a.n))
    return list_to_atoms(sal)


def make_elements(atoms):
    ret = OrderedDict()
    for n in atoms.n:
        name = atom.atomic_name(n)
        if name not in ret:
            ret[name] = 0
        ret[name] += 1
    return ret


def TF(boolean):
    if boolean:
        return 'T'
    else:
        return 'F'


def atoms_and_lattice_for_write(poscar, universal_scaling_factor=1.0, cartesian=True, sort=lambda x: x):
    atoms = sorted_atoms(poscar.atoms, sort=sort)
    lattice = poscar.lattice.lattice / universal_scaling_factor
    if cartesian:
        atoms.x = atoms.x / universal_scaling_factor
    else:
        atoms.x = cartesian_to_direct(
                atoms.x / universal_scaling_factor,
                poscar.lattice.reciprocal_lattice * universal_scaling_factor)
    return atoms, lattice


def write_comment(f, comment):
    f.write(comment + '\n')
    

def write_universal_scaling_factor(f, universal_scaling_factor):
    f.write("{:<019.14}\n".format(universal_scaling_factor))


def write_lattice(f, lattice):
    for lat in lattice:
        f.write(" {:<021.16} {:<021.16} {:<021.16}\n".format(*[l for l in lat]))


def write_element(f, atoms):
    element_dict = make_elements(atoms)
    for key in element_dict.keys():
        f.write(" {:>4}".format(key))
    f.write('\n')
    for val in element_dict.values():
        f.write(" {:>4}".format(val))
    f.write('\n')


def write_poscar(path, poscar, universal_scaling_factor=1.0, cartesian=True, sort=lambda x: x):
    atoms, lattice = atoms_and_lattice_for_write(
            poscar, universal_scaling_factor, cartesian, sort)
    with open(path, 'w') as f:
        write_comment(f, poscar.comment)
        write_universal_scaling_factor(f, universal_scaling_factor)
        write_lattice(f, lattice)
        write_element(f, atoms)
        if poscar.selective_dynamics_flags is not None:
            f.write("Selective dynamics\n")
        if cartesian:
            f.write("Cartesian\n")
        else:
            f.write("Direct\n")
        if poscar.selective_dynamics_flags is None:
            for x in atoms.x:
                f.write("{:< 020.14} {:< 020.14} {:< 020.14}\n".format(x[0], x[1], x[2]))
        else:
            for x, flag in zip(atoms.x, poscar.selective_dynamics_flags):
                f.write("{:< 020.14} {:< 020.14} {:< 020.14}   {}   {}   {}\n".format(
                    x[0], x[1], x[2], TF(flag[0]), TF(flag[1]), TF(flag[2])))
        f.write('\n')
