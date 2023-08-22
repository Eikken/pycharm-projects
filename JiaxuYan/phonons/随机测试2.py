#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   随机测试2.py    
@Time    :   2023/3/7 17:23  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
    time1 = time.time()
    time2 = time.time()
    print('>> Finished, use time %d s' % (time2 - time1))
'''

import numpy as np
import matplotlib.pyplot as plt

# Define constants
t = 2.8 # eV

# Define the Hamiltonian
H = np.array([[0, t], [t, 0]])

# Compute the eigenvalues
eigvals, eigvecs = np.linalg.eigh(H)

# Plot the band structure
k = np.linspace(-np.pi, np.pi, 100)

E1 = []
E2 = []

for kval in k:
    Hk = H + np.array([[0, np.exp(1j*kval)], [np.exp(-1j*kval), 0]])
    eigvalsk, eigvecsk = np.linalg.eigh(Hk)
    E1.append(eigvalsk[0])
    E2.append(eigvalsk[1])

plt.plot(k,E1)
plt.plot(k,E2)
plt.xlabel('k')
plt.ylabel('E')
plt.title('Bilayer Graphene Band Structure')
plt.show()
# # !/usr/bin/python
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 双层石墨烯声子色散系数
# gamma = 6.3
# # 声子能量
# omega = 0.2
# # Planck常数
# hbar = 6.582119514e-16
#
#
# # 计算声子色散
# def Dispersion(gamma, omega, hbar):
#     k = np.arange(-50, 50, 0.01)
#     E_ph = hbar * omega * np.sqrt(1 + (2.0 * gamma * k) ** 2)
#     return k, E_ph
#
#
# # 计算声子色散
# k, E_ph = Dispersion(gamma, omega, hbar)
#
# # 绘制声子色散图
# plt.plot(k, E_ph, 'k-', lw=2)
# # plt.xlim(-50,50)
# # plt.ylim(0,2.2)
# plt.xlabel('Wave vector k (1/Å)')
# plt.ylabel('Phonon energy (eV)')
# plt.title('Phonon dispersion of bilayer graphene')
# plt.show()
