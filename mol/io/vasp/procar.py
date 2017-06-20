import numpy as np
import pandas as pd


class Band(object):
    def __init__(self, i_band, energy, occ, site_projected_wave, phase_factor=None):
        self.i_band = i_band
        self.energy = energy
        self.occ = occ
        self.site_projected_wave = site_projected_wave
        self.phase_factor = phase_factor


class Kpoint(object):
    def __init__(self, i_kpoint, k, weight, bands):
        self.i_kpoint = i_kpoint
        self.k = k
        self.weight = weight
        self.bands = bands


def convert(string):
    striped = string.strip()
    s = striped[1:] if striped[0] == '-' else striped
    if s.isdigit():
        return int(striped)
    elif s.replace('.', '', 1).isdigit():
        return float(striped)
    else:
        return striped


def read_row(f):
    return [convert(s) for s in f.readline().split()]


def read_phase_q(line):
    #PROCAR new format
    #PROCAR lm decomposed
    #PROCAR lm decomposed + phase
    if "phase" in line:
        return True
    else:
        return False


def read_number_of_kpoints_bands_ions(f):
    # info['#', 'of', 'k-points:', '16', '#', 'of', 'bands:', '8', '#', 'of', 'ions:', '2']
    [_, _, _, n_kpoints, _, _, _, n_bands, _, _, _, n_ions] = read_row(f)
    return n_kpoints, n_bands, n_ions


def read_site_projected_wave(f, n_ions):
    columns = f.readline().split()
    data = [read_row(f) for _ in range(n_ions + 1)]
    return pd.DataFrame(columns=columns, data=data)


def read_complex_row(f):
    r1 = read_row(f)
    r2 = read_row(f)
    assert r1[0] == r2[0]
    ion = r1[0]
    return [ion] + [r + i * 1j for r, i in zip(r1[1:], r2[1:])]


def read_phase_factor(f, n_ions):
    columns = f.readline().split()
    data = [read_complex_row(f) for _ in range(n_ions)]
    return pd.DataFrame(columns=columns, data=data)


def read_band_energy_occ(f):
    #['band', 1, '#', 'energy', -11.85469113, '#', 'occ.', 2.0]
    [_, band, _, _, energy, _, _, occ] = read_row(f)
    return band, energy, occ


def read_band(f, n_ions):
    band, energy, occ = read_band_energy_occ(f)
    f.readline()
    site_projected_wave = read_site_projected_wave(f, n_ions)
    f.readline()
    return Band(band, energy, occ, site_projected_wave)


def read_band_with_phase_factor(f, n_ions):
    band, energy, occ = read_band_energy_occ(f)
    f.readline()
    site_projected_wave = read_site_projected_wave(f, n_ions)
    phase_factor = read_phase_factor(f, n_ions)
    f.readline()
    return Band(band, energy, occ, site_projected_wave, phase_factor)


def read_kpoint_weight(f):
    # k-point    1 :    0.00000000 0.00000000 0.00000000     weight = 0.00462963
    [_, i_kpoint, _, k1, k2, k3, _, _, weight] = read_row(f)
    return i_kpoint, np.array([k1, k2, k3]), weight


def read_kpoint(f, reader, n_bands, n_ions):
    f.readline()
    i_kpoint, k, weight = read_kpoint_weight(f)
    f.readline()
    bands = [reader(f, n_ions) for _ in range(n_bands)]
    return Kpoint(i_kpoint, k, weight, bands)


def read_procar(path):
    with open(path) as f:
        title = f.readline().strip()
        read_phase = read_phase_q(title)
        n_kpoints, n_bands, n_ions = read_number_of_kpoints_bands_ions(f)
        if read_phase:
            reader = read_band_with_phase_factor
        else:
            reader = read_band
        kpoints_alpha = [read_kpoint(f, reader, n_bands, n_ions) for _ in range(n_kpoints)]
        try:
            n_kpoints, n_bands, n_ions = read_number_of_kpoints_bands_ions(f)
        except ValueError:
            return (kpoints_alpha, )
        kpoints_beta = [read_kpoint(f, reader, n_bands, n_ions) for _ in range(n_kpoints)]
        return kpoints_alpha, kpoints_beta

def test():
    ks = read_procar("PROCAR")
    print(ks[0][0].bands[0].phase_factor)

if __name__ == '__main__':
    test()
