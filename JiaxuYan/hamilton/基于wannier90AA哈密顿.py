#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   基于wannier90AA哈密顿.py    
@Time    :   2023/4/16 16:45  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
    time1 = time.time()
    time2 = time.time()
    print('>> Finished, use time %d s' % (time2 - time1))
'''
import functools
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from JiaxuYan.hamilton.基于fang2015提取MoS2轨道 import make_path, get_sphere


def isinDirect(v_, hvts):
    for vs in hvts:
        if ''.join(v_) == ''.join(list(map(str, vs))):
            return True
    return False


def plot_rec(*args, **kwargs):
    av_ = args[0]
    av1_, av2_, av3_ = [args[0][i, :] for i in range(3)]
    bv1_, bv2_ = args[1], args[2]
    Gamma_ = np.array([0, 0])
    Mid_ = 1 / 2 * bv1_
    K1_ = 1 / 3 * (2 * bv1_ - bv2_)
    K2_ = -1 / 3 * (2 * bv1_ - bv2_)
    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes()
    ax.arrow(0, 0, av1_[0], av1_[1], length_includes_head=False, head_width=0.05, fc='b', ec='k')
    ax.arrow(0, 0, av2_[0], av2_[1], length_includes_head=False, head_width=0.05, fc='b', ec='k')
    ax.arrow(0, 0, bv1_[0], bv1_[1], length_includes_head=False, head_width=0.05, fc='r', ec='red')
    ax.arrow(0, 0, bv2_[0], bv2_[1], length_includes_head=False, head_width=0.05, fc='r', ec='red')
    ax.plot([Gamma_[0], Mid_[0], K1_[0], 0], [Gamma_[1], Mid_[1], K1_[1], 0])
    ax.scatter([Gamma_[0], Mid_[0], K1_[0], 0], [Gamma_[1], Mid_[1], K1_[1], 0])
    ax.set_xlim(-1, 4)
    ax.set_ylim(-2, 4)
    ax.grid()
    plt.show()


def phase(*args, **kwargs):
    rij = kwargs['hr']
    wk = kwargs['wk']
    R_vec = 0
    for ii in range(3):
        R_vec += rij[ii] * basis_vector[ii]
    inner_product = np.dot(R_vec, wk)
    return np.exp(1j * inner_product)


def Hamham(wak, tij, rij):
    h = np.zeros((nb, nb), dtype=complex)
    for ii in range(nb):
        for jj in range(nb):
            for vij in range(len(rij[ii][jj])):
                hr = rij[ii][jj][vij]
                h[ii, jj] = h[ii, jj] + tij[ii][jj][vij][0] * phase(hr=hr, wk=wak)
    return h


def read_dat(*args, **kwargs):
    with open("data/wannier90_hr_21.79.dat", "r") as f:
        lines = f.readlines()
    rij = [[[] for col in range(nb)] for row in range(nb)]
    tij = [[[] for col in range(nb)] for row in range(nb)]
    for line in lines:
        ll = line.split()
        if isinDirect(ll[:3], kwargs['hvts']):
            x, y, z, frAt, toAt = list(map(int, ll[:5]))
            t_real, t_image = list(map(float, ll[5:]))
            rij[frAt - 1][toAt - 1].append([x, y, z])
            tij[frAt - 1][toAt - 1].append([t_real + 1j * t_image])
    return rij, tij


def select22orbitals(orbi):
    hvts = get_sphere(orbi)
    path = kpath
    result = np.zeros([len(path), nb])  # 解的矩阵
    Rij, Tij = read_dat(hvts=hvts)
    print('read data!')
    hamilton = functools.partial(Hamham, tij=Tij, rij=Rij)
    idx = 0
    for kxy in range(len(path)):
        k = path[kxy]
        w, t = np.linalg.eig(hamilton(k))
        w = list(w)
        w.sort()
        result[idx, :] = np.real(w)  # 将本征值进行保存
        idx += 1
    xk = [0, 2, rt3 + 2, rt3 + 3, rt3 + 5]
    kk = np.linspace(0, 6.7, num=len(path))  # 作为x轴，使其和本征值矩阵每一列的y的值个数相同
    plt.figure(figsize=(9, 7), dpi=200)
    plt.plot(kk, result, linewidth=0.4, color="r")
    plt.xticks(xk, ["K", "Γ", "M", "K'", "Γ"])
    plt.xlim(0, rt3 + 5)
    plt.tick_params(labelsize=18)
    plt.ylabel("Energy(eV)", fontsize=14)
    plt.axvline(2, color='gray', linestyle='--')
    plt.axvline(rt3 + 2, color='gray', linestyle='--')
    plt.axvline(rt3 + 3, color='gray', linestyle='--')
    # plt.title('n=%d' % orbi)
    plt.savefig('png/orbitalAA/%d orbital.png' % orbi)
    print('finish %d/%d' % (orbi, 11))


def selectAllOrbital():
    path = kpath
    with open("data/wannier90_hr_21.79.dat", "r") as f:
        lines = f.readlines()
    rij = [[[] for col in range(nb)] for row in range(nb)]
    tij = [[[] for col in range(nb)] for row in range(nb)]
    for line in lines:
        ll = line.split()
        if len(ll) == 7:
            x, y, z, frAt, toAt = list(map(int, ll[:5]))
            t_real, t_image = list(map(float, ll[5:]))
            rij[frAt - 1][toAt - 1].append([x, y, z])
            tij[frAt - 1][toAt - 1].append([t_real + 1j * t_image])
    print('read data!')
    hamilton = functools.partial(Hamham, tij=tij, rij=rij)
    idx = 0
    result = np.zeros([len(path), nb])
    for kxy in range(len(path)):
        k = path[kxy]
        w, t = np.linalg.eig(hamilton(k))
        w = list(w)
        w.sort()
        result[idx, :] = np.real(w)  # 将本征值进行保存
        idx += 1
    xk = [0, 2, rt3 + 2, rt3 + 3, rt3 + 5]
    kk = np.linspace(0, 6.7, num=len(path))  # 作为x轴，使其和本征值矩阵每一列的y的值个数相同
    plt.figure(figsize=(9, 7), dpi=200)
    plt.plot(kk, result, linewidth=0.4, color="r")
    plt.xticks(xk, ["K", "Γ", "M", "K'", "Γ"])
    plt.ylabel("Energy(eV)", fontsize=10)
    plt.axvline(2, color='gray', linestyle='--')
    plt.axvline(rt3 + 2, color='gray', linestyle='--')
    plt.axvline(rt3 + 3, color='gray', linestyle='--')
    plt.title('n=all')
    plt.savefig('png/21-79/all orbital.png')
    plt.show()
    print('Saved all orbital figure!')


def select154orbitals(orbi):
    hvts = get_sphere(orbi)
    path = kpath
    result = np.zeros([len(path), nb])  # 解的矩阵
    Rij, Tij = read_dat(hvts=hvts)
    print('read data!')
    hamilton = functools.partial(Hamham, tij=Tij, rij=Rij)
    idx = 0
    for kxy in range(len(path)):
        k = path[kxy]
        w, t = np.linalg.eig(hamilton(k))
        w = list(w)
        w.sort()
        result[idx, :] = np.real(w)  # 将本征值进行保存
        idx += 1
    xk = [0, 2, rt3 + 2, rt3 + 3, rt3 + 5]
    kk = np.linspace(0, 6.7, num=len(path))  # 作为x轴，使其和本征值矩阵每一列的y的值个数相同
    plt.figure(figsize=(9, 7), dpi=200)
    plt.plot(kk, result, linewidth=0.4, color="r")
    plt.xticks(xk, ["K", "Γ", "M", "K'", "Γ"])
    plt.ylabel("Energy(eV)", fontsize=10)
    plt.axvline(2, color='gray', linestyle='--')
    plt.axvline(rt3 + 2, color='gray', linestyle='--')
    plt.axvline(rt3 + 3, color='gray', linestyle='--')
    # plt.title('n=%d' % orbi)
    plt.xlim(0, rt3 + 5)
    plt.tick_params(labelsize=18)
    plt.savefig('png/21-79/%d orbital.png' % orbi)
    # plt.show()
    print('finish %d/%d' % (orbi, 11))


if __name__ == '__main__':
    # start here
    a = 3.1592038768
    nb = 154
    rt3 = 3 ** 0.5
    pi = np.pi
    a1 = a * np.array([rt3 / 2, -1 / 2, 0])
    a2 = a * np.array([0, 1, 0])
    a3 = a * np.array([0, 0, 5])
    basis_vector = np.array([a1, a2, a3])

    b1 = 2 * pi / a * np.array([1 / rt3, 1, 0])
    b2 = 4 * pi / a / rt3 * np.array([1, 0, 0])
    # plot_rec(basis_vector, b1, b2)
    Gamma = np.array([0, 0, 0])
    Mid = 1 / 2 * b1
    K1 = 1 / 3 * (2 * b1 - b2)
    K2 = -1 / 3 * (2 * b1 - b2)
    kpath = make_path(K2, Gamma, Mid, K1, Gamma, step=0.05)
    # kpath = make_path(Gamma, Mid, K1, Gamma, step=0.05)

    # K G M K
    # selectAllOrbital()
    # for orb in range(2, 11):
    select154orbitals(orbi=3)
    # select22orbitals(orbi=3)