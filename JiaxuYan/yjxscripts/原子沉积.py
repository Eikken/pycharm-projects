#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   原子沉积.py    
@Time    :   2023/4/7 10:57  
@E-mail  :   iamwxyoung@qq.com
@Tips    :
首先要知道，范数是一个标量，它是对向量（或者矩阵）的度量
范数包含 0 范数、1范数、2范数........ P范数。
其中：
0 范数，表示向量中非零元素的个数。
1 范数，表示向量中各个元素绝对值之和。
2 范数，表示向量中各个元素平方和 的 1/2 次方，L2 范数又称 Euclidean 范数或者 Frobenius 范数。
p 范数，表示向量中各个元素绝对值 p 次方和 的 1/p 次方

'''

import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# ------------------------------------------------------------
# Define a hook class to deposit atoms from the reservoir
# onto the active surface of the substrate.
# ------------------------------------------------------------

class VaporDepositionHook:
    def __init__(self,
                 deposition_interval,
                 configuration,
                 substrate_atoms,
                 vapor_temperature=300.0 * Kelvin):
        """
        Constructor
        """
        # Store the parameters.

        # 沉积时间间隔
        self._deposition_interval = deposition_interval

        # 存储原始坐标，约束原子
        self._original_coordinates = configuration.cartesianCoordinates().inUnitsOf(Angstrom)

        # 非沉积原子
        self._substrate_atoms = substrate_atoms

        # 温度.
        self._vapor_temperature = vapor_temperature

        # 沉积指数.
        self._deposited_index = 0

        # 获取热源原子，也就是z轴较低的底层原子
        self._lowest_substrate = self._original_coordinates[self._substrate_atoms, 2].min()
        self._reservoir_indices = numpy.where(self._original_coordinates[:, 2] < self._lowest_substrate - 0.1)[
            0].tolist()

        # Get the lattice vectors.
        cell = configuration.bravaisLattice().primitiveVectors().inUnitsOf(Angstrom)
        self._lx = cell[0, 0]
        self._ly = cell[1, 1]
        self._lz = cell[2, 2]

    def __call__(self, step, time, configuration, forces, stress):
        """ Call the hook during MD. """
        # Get the coordinates.
        coordinates = configuration.cartesianCoordinates().inUnitsOf(Angstrom)

        # 所述 储层 由位于底物底层以下或在顶层以上的所有原子组成。
        self._reservoir_indices = numpy.where((coordinates[:, 2] < self._lowest_substrate - 0.1) |
                                              (coordinates[:, 2] > self._lz))[0]

        # 冻结储层原子并在每一步重置它们的位置。
        velocities = configuration.velocities()
        velocities[self._reservoir_indices, :] = 0.0 * Angstrom / fs
        configuration._changeAtoms(indices=self._reservoir_indices,
                                   positions=self._original_coordinates[self._reservoir_indices] * Angstrom)
        # 检查是否到了从储层中沉积新原子的时候。
        if (step % self._deposition_interval) == 0:
            # Get elements and velocities.
            elements = numpy.array(configuration.elements())
            velocities = configuration.velocities().inUnitsOf(Angstrom / fs)

            # 如果底层是空的，什么也不做。
            if len(self._reservoir_indices) == 0:
                return

            # 决定下一步存放哪个元素
            possible_elements = [Silicon, Carbon]
            element_index = self._deposited_index % 2
            next_element = possible_elements[element_index]
            reservoir_elements = elements[self._reservoir_indices]

            # 获取候选元素
            possible_atoms = numpy.where(reservoir_elements == next_element)[0]

            # 如果不可用，请尝试其他元素
            if len(possible_atoms) == 0:
                next_element = possible_elements[element_index - 1]
                possible_atoms = numpy.where(reservoir_elements == next_element)[0]

            # Return back the global atom indices.
            possible_atoms = self._reservoir_indices[possible_atoms]

            # Pick an atom at the bottom of the reservoir
            lowest_atom = numpy.argmin(coordinates[possible_atoms, 2])
            next_index = possible_atoms[lowest_atom]

            # 将沉积原子置于表面上方任意横向位置。
            new_coords = numpy.array([numpy.random.uniform() * self._lx,
                                      numpy.random.uniform() * self._ly,
                                      self._lz - 15.0])

            configuration._changeAtoms(indices=[next_index], positions=new_coords * Angstrom)
            wrap(configuration)

            # 根据麦克斯韦-玻尔兹曼定律，画出原子在特定温度下的随机速度。
            m = next_element.atomicMass()
            sig = self._vapor_temperature * boltzmann_constant / m
            sig = sig.inUnitsOf(Ang ** 2 / fs ** 2)
            sig = numpy.sqrt(sig)

            new_velocity = numpy.random.normal(scale=sig, size=3)
            velocity_norm = numpy.linalg.norm(new_velocity)  # 欧几里得2番薯
            # Set the velocity so that it points towards the active surface.
            new_velocity = numpy.array([0.0, 0.0, -velocity_norm])
            velocities[next_index] = new_velocity

            # Set the new velocities on the configuration.
            configuration.setVelocities(velocities * Angstrom / fs)

            self._deposited_index += 1

    # ------------------------------------------------------------
    # End of hook class definition
    # ------------------------------------------------------------

if __name__ == '__main__':
    # start here
    # Initialize the hook class.
    deposition_hook = VaporDepositionHook(deposition_interval=5000,
                                          configuration=bulk_configuration,
                                          substrate_atoms=substrate + bottom,
                                          vapor_temperature=2400.0 * Kelvin)
    