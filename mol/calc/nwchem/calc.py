import ase
from ase.calculators.nwchem import NWChem

def calc(n, x, xc='PBE', convergence={'energy': None, 'density': None, 'gradient' : None, 'lshift': None, 'damp': None}, mult=1, charge=0, maxiter=30):
    atoms = ase.Atoms(n, x.reshape([len(n), 3]))
    c = NWChem(xc=xc, convergence=convergence, mult=mult, charge=charge, maxiter=maxiter)
    atoms.set_calculator(c)
    energy = atoms.get_potential_energy()
    force = atoms.get_forces()
    del(atoms)
    del(c)
    return energy, -force.flatten()
