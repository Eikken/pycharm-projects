#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   能带图.py    
@Time    :   2020/12/30 13:42  
@Tips    :    
'''

import matplotlib.pyplot as plt
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.electronic_structure.plotter import BSDOSPlotter,BSPlotter,BSPlotterProjected,DosPlotter

bs_vasprun = Vasprun("vasprun.xml",parse_projected_eigen=True)
bs_data = bs_vasprun.get_band_structure(line_mode=True)
dos_data=bs_vasprun.complete_dos

banddos_fig = BSDOSPlotter(bs_projection=None, \
dos_projection=None, vb_energy_range=-5, \
fixed_cb_energy=5)
band_fig = BSPlotter(bs=bs_data)
band_fig.get_plot()
plt.ylim(-2,1)
