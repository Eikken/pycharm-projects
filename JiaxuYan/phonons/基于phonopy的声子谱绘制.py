#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   基于phonopy的声子谱绘制.py    
@Time    :   2023/4/11 14:31  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   

'''

import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from phonopy.units import AbinitToTHz
import sys
from phonopy import Phonopy
from phonopy.file_IO import parse_FORCE_CONSTANTS, parse_disp_yaml, parse_FORCE_SETS
from phonopy.structure.atoms import PhonopyAtoms
from ase.atoms import Atoms
from ase.io import read
from numpy.linalg import inv
from ase.dft.kpoints import *
from ase.build import bulk, mx2
from unfolding.phonon_unfolder import phonon_unfolder
from unfolding.plotphon import plot_band_weight


def print_data():
    for i in range(5):
        sys.stdout.write(str(i) + '\r')
        time.sleep(1)
        sys.stdout.flush()


def read_phonopy(sposcar='SPOSCAR', sc_mat=np.eye(3), force_constants=None,  disp_yaml=None, force_sets=None):
    atoms = read(carPath)
    fcs = parse_FORCE_CONSTANTS(forcePath)
    primitive_matrix=inv(sc_mat)

    bulk_ = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        scaled_positions=atoms.get_scaled_positions(),
        cell=atoms.get_cell())
    pho = Phonopy(
        bulk_,
        supercell_matrix=np.eye(3),
        primitive_matrix=primitive_matrix,
    )
    pho.set_force_constants(fcs)
    return pho


def unf(phonon, sc_mat, qpoints, knames=None, x=None, xpts=None):
    prim=phonon.get_primitive()
    prim=Atoms(symbols=prim.get_chemical_symbols(), cell=prim.get_cell(), positions=prim.get_positions())
    flg = 1

    sc_qpoints=np.array([np.dot(q, sc_mat) for q in qpoints])
    phonon.set_qpoints_phonon(sc_qpoints, is_eigenvectors=True)
    freqs, eigvecs = phonon.get_qpoints_phonon()
    uf=phonon_unfolder(atoms=prim, supercell_matrix=sc_mat, eigenvectors=eigvecs, qpoints=sc_qpoints, phase=False)
    weights = uf.get_weights()

    ax=plot_band_weight([list(x)]*freqs.shape[1], freqs.T*33.356, weights[:,:].T*0.99+0.001, width=5, xticks=[knames, xpts], style='alpha')
    print("precessing %d/%d" % (flg, 100), end='\r', flush=True)
    flg += 1
    return ax


def phonopy_unfold(sc_mat=np.diag([1,1,1]), unfold_sc_mat=np.diag([3,3,3]),force_constants='FORCE_CONSTANTS', sposcar='SPOSCAR', qpts=None, qnames=None, xqpts=None, Xqpts=None):
    phonon=read_phonopy(sc_mat=sc_mat, force_constants=force_constants, sposcar=sposcar)
    print('read phonopy!')
    ax=unf(phonon, sc_mat=unfold_sc_mat, qpoints=qpts, knames=qnames,x=xqpts, xpts=Xqpts )
    return ax


if __name__ == '__main__':

    time1 = time.time()

    forcePath = r'data/FORCE_CONSTANTS_sposcar'
    carPath = r'data/SPOSCAR_fcs'

    # atoms = bulk('MoS2', 'hcp', a=3.18)
    atoms = mx2(formula='MoS2', kind='2H', a=3.18, thickness=3.19, size=(1, 1, 1), vacuum=None)
    points = get_special_points('hcp', atoms.cell, eps=0.01)
    path_highsym = [points[k] for k in 'GKMG']
    kpath = bandpath(path_highsym, atoms.cell, 100)
    kpts, x, X = bandpath(path_highsym, atoms.cell, 300)
    names = ['$\Gamma$', 'K', 'M', '$\Gamma$']
    # path = bandpath(...)
    # kpts = path.kpts
    # (x, X, labels) = path.get_linear_kpoint_axis()
    # here is the unfolding. Here is an example of 3*3*3 fcc cell.
    ax = phonopy_unfold(sc_mat=np.diag([1, 1, 1]),  # supercell matrix for phonopy.
                        unfold_sc_mat=np.diag([10, 10, 1]),  # supercell matrix for unfolding
                        force_constants=forcePath,  # FORCE_CONSTANTS_1213 file path
                        sposcar=carPath,  # SPOSCAR_fcs file for phonopy
                        qpts=kpts,  # q-points. In primitive cell!
                        qnames=names,  # Names of high symmetry q-points in the q-path.
                        xqpts=x,  # kpath.get_linear_kpoint_axis(),  # x-axis, should have the same length of q-points.
                        Xqpts=X  # x-axis
                        )
    plt.savefig('png/MoS2 10X10.png')
    plt.show()

    time2 = time.time()
    print('>> Finished, use time %.2f min' % ((time2 - time1)/60))