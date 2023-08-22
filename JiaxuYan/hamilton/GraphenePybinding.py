# Import libraries
import pybinding as pb
import matplotlib.pyplot as plt
import numpy as np
from math import pi
from pybinding.repository import graphene

# create a lattice in format of pyBinding
def GetLattice(onsite_energy=[-6.72160353, 7.02294188, -0.10201043, 6.30046142, -6.72160353, 7.02294188, -0.10201043,
                              6.30046142]):
    lattice = pb.Lattice(
        a1=[1.23435000, 2.14452000, 0.00000000],
        a2=[1.23435000, -2.14452000, 0.00000000],
        # a3=[0.00000000, 0.00000000, 20.00000000]
    )

    lattice.add_sublattices(
        # name and positions
        ('o1:C:1:s', [0.00414000, -0.00486300, 10.00000000], onsite_energy[0]),
        ('o2:C:1:p_y', [0.00414000, -0.00486300, 10.00000000], onsite_energy[1]),
        ('o3:C:1:p_z', [0.00414000, -0.00486300, 10.00000000], onsite_energy[2]),
        ('o4:C:1:p_x', [0.00414000, -0.00486300, 10.00000000], onsite_energy[3]),
        ('o5:C:1:s', [1.23861100, -0.72006000, 10.00000000], onsite_energy[4]),
        ('o6:C:1:p_y', [1.23861100, -0.72006000, 10.00000000], onsite_energy[5]),
        ('o7:C:1:p_z', [1.23861100, -0.72006000, 10.00000000], onsite_energy[6]),
        ('o8:C:1:p_x', [1.23861100, -0.72006000, 10.00000000], onsite_energy[7])
    )

    lattice.add_hoppings(

        # between main cell and the cell (1,1,0)
        ([1, 1, 0], 'o5:C:1:s', 'o1:C:1:s', -4.21142108),
        ([1, 1, 0], 'o5:C:1:s', 'o2:C:1:p_y', -2.61105392),
        ([1, 1, 0], 'o5:C:1:s', 'o4:C:1:p_x', -4.50594517),
        ([1, 1, 0], 'o6:C:1:p_y', 'o1:C:1:s', 2.61105392),
        ([1, 1, 0], 'o6:C:1:p_y', 'o2:C:1:p_y', 0.31646459),
        ([1, 1, 0], 'o6:C:1:p_y', 'o4:C:1:p_x', 4.86600048),
        ([1, 1, 0], 'o7:C:1:p_z', 'o3:C:1:p_z', -2.50323009),
        ([1, 1, 0], 'o8:C:1:p_x', 'o1:C:1:s', 4.50594517),
        ([1, 1, 0], 'o8:C:1:p_x', 'o2:C:1:p_y', 4.86600048),
        ([1, 1, 0], 'o8:C:1:p_x', 'o4:C:1:p_x', 5.89411905),

        # between main cell and the cell (1,0,0)

        # between main cell and the cell (1,-1,0)

        # between main cell and the cell (0,1,0)
        ([0, 1, 0], 'o5:C:1:s', 'o1:C:1:s', -4.21142108),
        ([0, 1, 0], 'o5:C:1:s', 'o2:C:1:p_y', 5.20779649),
        ([0, 1, 0], 'o5:C:1:s', 'o4:C:1:p_x', 0.00044087),
        ([0, 1, 0], 'o6:C:1:p_y', 'o1:C:1:s', -5.20779649),
        ([0, 1, 0], 'o6:C:1:p_y', 'o2:C:1:p_y', 8.71381365),
        ([0, 1, 0], 'o6:C:1:p_y', 'o4:C:1:p_x', 0.00094958),
        ([0, 1, 0], 'o7:C:1:p_z', 'o3:C:1:p_z', -2.50323009),
        ([0, 1, 0], 'o8:C:1:p_x', 'o1:C:1:s', -0.00044087),
        ([0, 1, 0], 'o8:C:1:p_x', 'o2:C:1:p_y', 0.00094958),
        ([0, 1, 0], 'o8:C:1:p_x', 'o4:C:1:p_x', -2.50323001),

        # inside the main cell
        ([0, 0, 0], 'o1:C:1:s', 'o5:C:1:s', -4.21142108),
        ([0, 0, 0], 'o1:C:1:s', 'o6:C:1:p_y', 2.61067070),
        ([0, 0, 0], 'o1:C:1:s', 'o8:C:1:p_x', -4.50616721),
        ([0, 0, 0], 'o2:C:1:p_y', 'o5:C:1:s', -2.61067070),
        ([0, 0, 0], 'o2:C:1:p_y', 'o6:C:1:p_y', 0.31563697),
        ([0, 0, 0], 'o2:C:1:p_y', 'o8:C:1:p_x', -4.86552606),
        ([0, 0, 0], 'o3:C:1:p_z', 'o7:C:1:p_z', -2.50323009),
        ([0, 0, 0], 'o4:C:1:p_x', 'o5:C:1:s', 4.50616721),
        ([0, 0, 0], 'o4:C:1:p_x', 'o6:C:1:p_y', -4.86552606),
        ([0, 0, 0], 'o4:C:1:p_x', 'o8:C:1:p_x', 5.89494667)
    )

    return lattice


def make_k_path(k1, k2, step=0.01, **kwargs):
    # either choose num_steps or step
    num_steps = 1
    if 'num_steps' in kwargs:
        num_steps = kwargs['num_steps']
    else:
        num_steps = int(np.linalg.norm(k2 - k1) // step)

    # k_path.shape == num_steps, k_space_dimensions
    k_path = np.array([np.linspace(s, e, num_steps, endpoint=False)
                       for s, e in zip(k1, k2)]).T
    return k_path


# setup lattice with on-site potential terms
lat = GetLattice()

plt.figure()
lat.plot()

plt.show()

# create a periodic model, which generates the Hamiltonian from the lattice
model = pb.Model(lat, pb.translational_symmetry())

a_ = 2.14452000
lat_acc = a_/3**0.5*2/3**0.5

Gamma = [0, 0]
M = [0, 2 * np.pi / (3 * lat_acc)]
K2 = [2 * np.pi / (3 * 3 ** 0.5 * lat_acc), 2 * np.pi / (3 * lat_acc)]
K1 = [-4 * np.pi / (3 * 3 ** 0.5 * lat_acc), 0]

path_list_1 = [r'$\Gamma$', 'M', 'K', r'$\Gamma$']
path_1 = pb.make_path(Gamma, M, K2, Gamma, step=0.01)


bands_list = []
for k in path_1:
    model.set_wave_vector(k)
    solver = pb.solver.lapack(model)
    bands_list.append(solver.eigenvalues)

bands = pb.results.Bands(path_1, np.vstack(bands_list))
bands.plot(point_labels=path_list_1)
plt.show()


model.lattice.plot_brillouin_zone()
bands.plot_kpath()
plt.show()

