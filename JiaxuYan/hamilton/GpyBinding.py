# Import libraries
import pybinding as pb
import matplotlib.pyplot as plt
import numpy as np
from math import pi


# create a lattice in format of pyBinding
def GetLattice(onsite_energy=[-7.36665121, 8.60251220, 0.12697323, 7.55703943, -7.36665121, 8.60251220, 0.12697323,
                              7.55703943]):
    lat = pb.Lattice(
        a1=[1.23435000, 2.14452000, 0.00000000],
        a2=[1.23435000, -2.14452000, 0.00000000],
        # a3 = [0.00000000, 0.00000000, 20.00000000]
    )

    lat.add_sublattices(
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

    lat.add_hoppings(

        # between main cell and the cell (1,1,0)
        ([1, 1, 0], 'o5:C:1:s', 'o1:C:1:s', -4.02692322),
        ([1, 1, 0], 'o5:C:1:s', 'o2:C:1:p_y', -2.53772030),
        ([1, 1, 0], 'o5:C:1:s', 'o4:C:1:p_x', -4.37939195),
        ([1, 1, 0], 'o6:C:1:p_y', 'o1:C:1:s', 2.53772030),
        ([1, 1, 0], 'o6:C:1:p_y', 'o2:C:1:p_y', 0.43838920),
        ([1, 1, 0], 'o6:C:1:p_y', 'o4:C:1:p_x', 5.37419142),
        ([1, 1, 0], 'o7:C:1:p_z', 'o3:C:1:p_z', -2.67578619),
        ([1, 1, 0], 'o8:C:1:p_x', 'o1:C:1:s', 4.37939195),
        ([1, 1, 0], 'o8:C:1:p_x', 'o2:C:1:p_y', 5.37419142),
        ([1, 1, 0], 'o8:C:1:p_x', 'o4:C:1:p_x', 6.59855766),

        # between main cell and the cell (1,0,0)

        # between main cell and the cell (1,-1,0)

        # between main cell and the cell (0,1,0)
        ([0, 1, 0], 'o5:C:1:s', 'o1:C:1:s', -4.02692322),
        ([0, 1, 0], 'o5:C:1:s', 'o2:C:1:p_y', 5.06153119),
        ([0, 1, 0], 'o5:C:1:s', 'o4:C:1:p_x', 0.00042849),
        ([0, 1, 0], 'o6:C:1:p_y', 'o1:C:1:s', -5.06153119),
        ([0, 1, 0], 'o6:C:1:p_y', 'o2:C:1:p_y', 9.71273296),
        ([0, 1, 0], 'o6:C:1:p_y', 'o4:C:1:p_x', 0.00104876),
        ([0, 1, 0], 'o7:C:1:p_z', 'o3:C:1:p_z', -2.67578619),
        ([0, 1, 0], 'o8:C:1:p_x', 'o1:C:1:s', -0.00042849),
        ([0, 1, 0], 'o8:C:1:p_x', 'o2:C:1:p_y', 0.00104876),
        ([0, 1, 0], 'o8:C:1:p_x', 'o4:C:1:p_x', -2.67578610),

        # inside the main cell
        ([0, 0, 0], 'o1:C:1:s', 'o5:C:1:s', -4.02692322),
        ([0, 0, 0], 'o1:C:1:s', 'o6:C:1:p_y', 2.53734784),
        ([0, 0, 0], 'o1:C:1:s', 'o8:C:1:p_x', -4.37960776),
        ([0, 0, 0], 'o2:C:1:p_y', 'o5:C:1:s', -2.53734784),
        ([0, 0, 0], 'o2:C:1:p_y', 'o6:C:1:p_y', 0.43747515),
        ([0, 0, 0], 'o2:C:1:p_y', 'o8:C:1:p_x', -5.37366745),
        ([0, 0, 0], 'o3:C:1:p_z', 'o7:C:1:p_z', -2.67578619),
        ([0, 0, 0], 'o4:C:1:p_x', 'o5:C:1:s', 4.37960776),
        ([0, 0, 0], 'o4:C:1:p_x', 'o6:C:1:p_y', -5.37366745),
        ([0, 0, 0], 'o4:C:1:p_x', 'o8:C:1:p_x', 6.59947171)
    )

    return lat


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
plt.subplot(121)
plt.title('Lattice: xy')
lat.plot()

plt.subplot(122)
plt.title('Lattice: yz')
lat.plot(axes='yz')
plt.show()

# create a periodic model, which generates the Hamiltonian from the lattice
model = pb.Model(lat, pb.translational_symmetry())

# take corners of the BZ
b1, b2 = model.lattice.reciprocal_vectors()
# Example of significant points (here, user should define the path as a function of b1, b2 and b3)
# P1 = b1[0:2] / 2.0
# Gamma = np.array([0, 0])
# P2 = np.array([0.54*np.pi, 0])
a_ = 2.14452000
lat_acc = a_ / 3 ** 0.5 * 2 / 3 ** 0.5

Gamma = [0, 0]
M = [0, 2 * np.pi / (3 * lat_acc)]
K2 = [2 * np.pi / (3 * 3 ** 0.5 * lat_acc), 2 * np.pi / (3 * lat_acc)]
K1 = [-4 * np.pi / (3 * 3 ** 0.5 * lat_acc), 0]

# make a path between each of the two points
# kp1 = make_k_path(P1, Gamma, num_steps=50)
# kp2 = make_k_path(Gamma, P2, num_steps=50)
# kp3 = make_k_path(P2, P1, num_steps=50)
path_list_1 = [r'$\Gamma$', 'M', 'K', r'$\Gamma$']
path_1 = pb.make_path(Gamma, M, K2, Gamma, step=0.01)
# define indexes
# point_indices = [0, kp1.shape[0] - 1, kp1.shape[0] + kp2.shape[0] - 1, kp1.shape[0] + kp2.shape[0] + kp3.shape[0] - 1]
# full_kpath = pb.results.Path(np.vstack((kp1, kp2, kp3)), point_indices=point_indices)

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
