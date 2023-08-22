import numpy as np

from ase.dft.kpoints import *
import matplotlib.pyplot as plt
from unfolding.phonopy_unfolder import phonopy_unfold

def run_unfolding():

    # Generate k-path for FCC structure
    from ase.build import bulk
    atoms = bulk('C', 'hcp', a=3.172129999999999)
    points = get_special_points('hcp', atoms.cell, eps=0.01)
    path_highsym = [points[k] for k in 'GKMG']
    kpts, x, X = bandpath(path_highsym, atoms.cell, 300)
    names = ['$\Gamma$', 'K', 'M','$\Gamma$']


    # here is the unfolding. Here is an example of 3*3*3 fcc cell. 
    ax=phonopy_unfold(sc_mat=np.diag([1,1,1]),      # supercell matrix for phonopy.
                          unfold_sc_mat=np.diag([20,20,1]),   # supercell matrix for unfolding
                          force_constants='FORCE_CONSTANTS_1213', # FORCE_CONSTANTS_1213 file path
                          sposcar='SPOSCAR_fcs',                 # SPOSCAR_fcs file for phonopy
                          qpts=kpts,                         # q-points. In primitive cell!
                          qnames=names,                      # Names of high symmetry q-points in the q-path.
                          xqpts=x,                           # x-axis, should have the same length of q-points.
                          Xqpts=X                            # x-axis
    )
    plt.show()

run_unfolding()
