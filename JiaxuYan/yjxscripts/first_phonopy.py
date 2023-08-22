#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   first_phonopy.py    
@Time    :   2023/2/27 0:54  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
    time1 = time.time()
    time2 = time.time()
    print('>> Finished, use time %d s' % (time2 - time1))
'''

import time
import matplotlib.pyplot as plt
from phonopy.units import AbinitToTHz
import numpy as np
from phonopy import Phonopy
import phonopy as ph
from phonopy.structure.atoms import PhonopyAtoms

help(ph.load)
a = 5.404
unitcell = PhonopyAtoms(symbols=['Si'] * 8,
                        cell=(np.eye(3) * a),
                        scaled_positions=[[0, 0, 0],
                                          [0, 0.5, 0.5],
                                          [0.5, 0, 0.5],
                                          [0.5, 0.5, 0],
                                          [0.25, 0.25, 0.25],
                                          [0.25, 0.75, 0.75],
                                          [0.75, 0.25, 0.75],
                                          [0.75, 0.75, 0.25]])
phonon = Phonopy(unitcell,
                 supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
                 primitive_matrix=[[0, 0.5, 0.5],
                                   [0.5, 0, 0.5],
                                   [0.5, 0.5, 0]],
                 factor=AbinitToTHz)
phonon.generate_displacements(distance=0.03)
supercells = phonon.supercells_with_displacements
# phonon.forces = phonon.produce_force_constants()