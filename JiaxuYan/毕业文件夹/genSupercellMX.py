#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   genSupercellMX.py    
@Time    :   2022/9/15 10:53  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   扩胞技术参考graphene
            20220916完善了扩胞中的一些bug，现在可以直接输入参数extend 产生超胞了.
            20220919着重解决为什么band是平的，3*3*3=27个本征值为什么是排序好的从小到大
'''
import re
import time

import math

import matplotlib.pyplot as plt
import pybinding as pb
import pandas as pd
import numpy as np
import pybinding.repository.group6_tmd as g6tmd
from networkx import radius
from scipy.spatial import distance


def initial_gen_method(ex=1, **kwargs):
    name_ = kwargs['name']
    params = g6tmd._default_3band_params.copy()
    a, eps1, eps2, t0, t1, t2, t11, t12, t22 = params[name_]

    if ex == 1:
        return g6tmd.monolayer_3band(name_), a

    # strategy 不变a1 a2, 内部的原子坐标乘以分数坐标而来

    rt3 = math.sqrt(3)  # convenient constant
    r1 = [a, 0]
    r2 = [1 / 2 * a, rt3 / 2 * a]

    super_x = ex
    super_y = ex
    super_z = 1

    extendMatrix = np.array([[super_x, 0],
                             [0, super_y]])
    init_lattice = np.array([r1, r2])

    extendLattice = np.dot(init_lattice, extendMatrix)
    # print(extendLattice)
    # Mo = [0, 0]
    numMo = (ex + 1) ** 2
    frac = 0.0  # 分数坐标位置
    allMo = []

    for i in range(super_x):
        for j in range(super_y):
            allMo.append([(frac + j) / super_x, (frac + i) / super_y])
    newMoSet = np.dot(np.array(allMo), extendLattice)

    # print(newMoSet)  # a1 a2 不变，新的new_a1 new_a2 变大 把atom position add_sublattice
    # plt.plot(newMoSet[:, 0], newMoSet[:, 1], marker='.')
    # plt.show()

    new_a1 = extendLattice[0]  # [0.638      0.        ]
    new_a2 = extendLattice[1]  # [0.319      0.55252421]

    h1 = [[t0, -t1, t2],
          [t1, t11, -t12],
          [t2, t12, t22]]

    h2 = [[t0, 1 / 2 * t1 + rt3 / 2 * t2, rt3 / 2 * t1 - 1 / 2 * t2],
          [-1 / 2 * t1 + rt3 / 2 * t2, 1 / 4 * t11 + 3 / 4 * t22, rt3 / 4 * (t11 - t22) - t12],
          [-rt3 / 2 * t1 - 1 / 2 * t2, rt3 / 4 * (t11 - t22) + t12, 3 / 4 * t11 + 1 / 4 * t22]]

    h3 = [[t0, -1 / 2 * t1 - rt3 / 2 * t2, rt3 / 2 * t1 - 1 / 2 * t2],
          [1 / 2 * t1 - rt3 / 2 * t2, 1 / 4 * t11 + 3 / 4 * t22, rt3 / 4 * (t22 - t11) + t12],
          [-rt3 / 2 * t1 - 1 / 2 * t2, rt3 / 4 * (t22 - t11) - t12, 3 / 4 * t11 + 1 / 4 * t22]]

    h1 = np.array(h1)
    h2 = np.array(h2)
    h3 = np.array(h3)

    this_lat = pb.Lattice(a1=new_a1, a2=new_a2)

    metal_name, chalcogenide_name = re.findall("[A-Z][a-z]*", name_)  # Mo S
    mo = str(metal_name)
    for p in range(len(newMoSet)):
        this_lat.add_one_sublattice(mo + str(p), newMoSet[p], [eps1, eps2, eps2])  # , alias='Mo0')
        # alias用于设置原子属性与指定原子一致
    # H1
    this_lat.add_hoppings(([0, 0], 'Mo0', 'Mo1', h1))
    this_lat.add_hoppings(([0, 0], 'Mo2', 'Mo3', h1))

    this_lat.add_hoppings(([1, 0], 'Mo1', 'Mo0', h1))
    this_lat.add_hoppings(([1, 0], 'Mo3', 'Mo2', h1))

    this_lat.add_hoppings(([1, 0], 'Mo3', 'Mo0', h3))

    # H2
    this_lat.add_hoppings(([0, 0], 'Mo2', 'Mo0', h2))
    this_lat.add_hoppings(([0, 0], 'Mo3', 'Mo1', h2))

    this_lat.add_hoppings(([0, -1], 'Mo0', 'Mo2', h2))
    this_lat.add_hoppings(([0, -1], 'Mo1', 'Mo3', h2))

    this_lat.add_hoppings(([0, -1], 'Mo0', 'Mo3', h3))


    # H3
    this_lat.add_hoppings(([1, -1], 'Mo1', 'Mo2', h3))

    # other
    this_lat.add_hoppings(([0, 0], 'Mo2', 'Mo1', h3))

    return this_lat, a


def gen_bigger_lat(ex=1, **kwargs):
    name_ = kwargs['name']
    params = g6tmd._default_3band_params.copy()
    a, eps1, eps2, t0, t1, t2, t11, t12, t22 = params[name_]

    if ex == 1:
        return g6tmd.monolayer_3band(name_), a

    # strategy 不变a1 a2, 内部的原子坐标乘以分数坐标而来

    rt3 = math.sqrt(3)  # convenient constant
    r1 = [a, 0]
    r2 = [1 / 2 * a, rt3 / 2 * a]

    super_x = ex
    super_y = ex
    super_z = 1

    extendMatrix = np.array([[super_x, 0],
                             [0, super_y]])
    init_lattice = np.array([r1, r2])

    extendLattice = np.dot(init_lattice, extendMatrix)
    # print(extendLattice)
    # Mo = [0, 0]
    numMo = (ex + 1) ** 2
    frac = 0.0  # 分数坐标位置
    allMo = []

    for i in range(super_x):
        for j in range(super_y):
            allMo.append([(frac + j) / super_x, (frac + i) / super_y])
    newMoSet = np.dot(np.array(allMo), extendLattice)

    # print(newMoSet)  # a1 a2 不变，新的new_a1 new_a2 变大 把atom position add_sublattice
    # plt.plot(newMoSet[:, 0], newMoSet[:, 1], marker='.')
    # plt.show()

    new_a1 = extendLattice[0]  # [0.638      0.        ]
    new_a2 = extendLattice[1]  # [0.319      0.55252421]

    h1 = [[t0, -t1, t2],
          [t1, t11, -t12],
          [t2, t12, t22]]

    h2 = [[t0, 1 / 2 * t1 + rt3 / 2 * t2, rt3 / 2 * t1 - 1 / 2 * t2],
          [-1 / 2 * t1 + rt3 / 2 * t2, 1 / 4 * t11 + 3 / 4 * t22, rt3 / 4 * (t11 - t22) - t12],
          [-rt3 / 2 * t1 - 1 / 2 * t2, rt3 / 4 * (t11 - t22) + t12, 3 / 4 * t11 + 1 / 4 * t22]]

    h3 = [[t0, -1 / 2 * t1 - rt3 / 2 * t2, rt3 / 2 * t1 - 1 / 2 * t2],
          [1 / 2 * t1 - rt3 / 2 * t2, 1 / 4 * t11 + 3 / 4 * t22, rt3 / 4 * (t22 - t11) + t12],
          [-rt3 / 2 * t1 - 1 / 2 * t2, rt3 / 4 * (t22 - t11) - t12, 3 / 4 * t11 + 1 / 4 * t22]]

    h1 = np.array(h1)
    h2 = np.array(h2)
    h3 = np.array(h3)

    this_lat = pb.Lattice(a1=new_a1, a2=new_a2)

    # 下面用于手写扩胞hopping
    # this_lat.add_hoppings(([1, 0], 'Mo3', 'Mo0', h3))

    metal_name, chalcogenide_name = re.findall("[A-Z][a-z]*", name_)  # Mo S
    mo = str(metal_name)
    # print(mo + '1')
    # this_lat.add_one_sublattice('Mo0', newMoSet[0], [eps1, eps2, eps2])
    # atom_dict['Mo0'] = newMoSet[0]
    for p in range(len(newMoSet)):
        # print(mo + str(p+1), newMoSet[p], [eps1, eps2, eps2])
        # this_lat.add_one_alias(mo + str(p), 'Mo0', newMoSet[p])
        this_lat.add_one_sublattice(mo + str(p), newMoSet[p], [eps1, eps2, eps2])  # , alias='Mo0')
        # alias用于设置原子属性与指定原子一致
        # atom_dict[mo + str(p)] = newMoSet[p]  # {'Mo0':[0, 0, 0], etc.}

    this_lat.register_hopping_energies({
        'H1': h1,
        'H2': h2,  # [eV]
        'H3': h3,  # [eV]
    })
    # combinations_list = list(combinations(list(atom_dict.keys()), 2))
    # 单纯的距离算法已经不能满足判定条件，还要判断传递方向，例如3倍扩胞下，使用距离判断后hopping关系时
    # a1、a2等正负方向会产生某些预期之外的bug。

    # 根据0,1 1,0 1,-1 等传递关系附h123
    for i in range(super_x):
        for j in range(super_y - 1):
            # [0, 0] h1, h2, 方向
            m1 = mo + (ex * i + j).__str__()
            n1 = mo + (ex * i + j + 1).__str__()
            # h1
            print('h1 >> ', m1, n1, end='\t')
            this_lat.add_hoppings(([0, 0], m1, n1, h1))

            n2 = mo + (ex * j + i).__str__()
            m2 = mo + (ex * (j + 1) + i).__str__()
            # h2
            print('  h2 >> ', m2, n2)
            # this_lat.add_hoppings(([0, 0], m2, n2, -1 * h2))
            this_lat.add_hoppings(([0, 0], m2, n2, h2))

    for i in range(ex):  # [0, -1] h2; [1, 0] h1 分几种类型，见上面2X2方式
        # 1>>0; 3>>2
        m3 = mo + i.__str__()
        n3 = mo + (ex * (ex - 1) + i).__str__()
        this_lat.add_hoppings(([0, -1], m3, n3, h2))
        print('h1 m3 n3', m3, n3)

        # 0>>2;1>>3
        m4 = mo + (ex*(i+1)-1).__str__()
        n4 = mo + (ex * i).__str__()
        this_lat.add_hoppings(([1, 0], m4, n4, h1))
        print('h2 m4 n4', m4, n4)

    for i in range(ex-1):  # [0, -1], [1, 0], h3
        # 1>>0; 3>>2
        m5 = mo + i.__str__()
        n5 = mo + (ex * (ex - 1) + i + 1).__str__()
        this_lat.add_hoppings(([0, -1], m5, n5, h3))
        print('h3 m5 n5', m5, n5)

        # 0>>2;1>>3
        m6 = mo + (ex*(i+2)-1).__str__()
        n6 = mo + (ex * i).__str__()
        this_lat.add_hoppings(([1, 0], m6, n6, h3))
        print('h3 m6 n6', m6, n6)

    this_diag = ex * (ex - 1)
    for i in range(ex - 1):
        # 主对角线[0, 0] h3
        m7diag = mo + ((ex-i)*(ex-1)).__str__()
        n7diag = mo + ((ex-(i+1))*(ex-1)).__str__()
        this_lat.add_hoppings(([0, 0], m7diag, n7diag, h3))
        print('h3 m7diag n7diag', m7diag, n7diag)

    if ex - 2 > 0:
        # for i in range(sum(range(ex-1))*2):
        # sum(range(ex-1))*2 : a1-a2方向有这么多条，在ex-2>0时
        # 这里是处理3+以上的扩胞是[0,0] h3方向的hopping 比较复杂
        for i in range(ex - 2):
            for j in range(i+1):
                m7down = mo + (ex * (i + 1) - (ex-1)*j).__str__()
                n7down = mo + (ex * i + 1 - (ex-1)*j).__str__()
                print('m7down, n7down ', m7down, n7down)
                this_lat.add_hoppings(([0, 0], m7down, n7down, h3))
                m7up = mo + (ex ** 2 - (2 + i) - (ex - 1) * j).__str__()  # 2+i 用用于从后往前， (ex-1) * j 用于斜向落差值
                n7up = mo + (ex ** 2 - (2 + i) - (ex - 1) * (j + 1)).__str__()
                print('m7up, n7up ', m7up, n7up)
                this_lat.add_hoppings(([0, 0], m7up, n7up, h3))

    # 最后是 [1, -1] 的单独h3传递
    m7 = mo + (ex-1).__str__()
    n7 = mo + (ex**2-ex).__str__()
    this_lat.add_hoppings(([1, -1], m7, n7, h3))

    return this_lat, a


def path_integral(lat, acc):
    # path 1 : Gamma >> K >> M >> Gamma
    # path 2 : K1 >> Gamma >> M >> K2
    path_list_1 = [r'$\Gamma$', 'K', 'M', r'$\Gamma$']
    path_list_2 = ['K1', r'$\Gamma$', 'M', 'K2']

    # this_model = pb.Model(lat,
    #                       # pb.translational_symmetry(),
    #                       pb.circle(radius=1.60)
    #                       )
    #
    # this_model.plot()
    # plt.title('extend lattice')
    # plt.axis('off')
    # plt.show()
    # plt.clf()

    this_model = pb.Model(lat,
                          pb.translational_symmetry(),
                          # pb.circle(radius=2)
                          )
    plt.rcParams.update({'font.size': 18})
    this_model.lattice.plot()
    # plt.title('extend lattice')
    plt.tick_params(labelsize=18)
    plt.axis('off')
    plt.show()
    plt.clf()

    k_points = this_model.lattice.brillouin_zone()
    Gamma = [0, 0]
    K = k_points[1]  # convenient get K points reciprocal vector
    M = (k_points[0] + k_points[1]) / 2
    K2 = k_points[2]

    this_path = pb.results.make_path(Gamma, K, M, Gamma, step=0.1)
    # this_path = pb.results.make_path(K, M, Gamma, K2, step=0.1)

    bands_list = []
    for k in this_path:
        this_model.set_wave_vector(k)
        solver = pb.solver.lapack(this_model)
        bands_list.append(solver.eigenvalues)

    bands = pb.results.Bands(this_path, np.vstack(bands_list))

    plt.figure(figsize=(5, 4), dpi=100)
    # plt.subplot(121, title="%dX%d path band plot"%(extend, extend))
    # bands.plot(point_labels=path_list_1)
    kp = bands.k_path
    print(len(kp))
    eg = bands.energy
    xkp = np.arange(len(kp))
    plt.plot(xkp, eg, lw=1)
    # for i in range(28):
    #     if i % 2 == 0:
    #         plt.plot(xkp, eg[:, i], lw=1, color='blue')
    #     else:
    #         plt.plot(xkp, eg[:, i], lw=1, color='green')
    fontSz = 18
    plt.axhline(y=0, linestyle='--', color='gray', linewidth=0.5)
    ax = plt.gca()  # 获取边框
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    xk = [0, int(len(kp)*2/4.7), int(len(kp)*3/4.7), len(kp)]
    plt.xticks(xk,  [r'$\Gamma$', 'K', 'M', r'$\Gamma$'])
    # xk = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12]
    # plt.yticks(xk, [0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.xlim(0, len(kp))
    plt.axvline(xk[1], color='gray', linestyle='--', linewidth=1, label='K')
    plt.axvline(xk[2], color='gray', linestyle='--', linewidth=1, label='K')
    plt.ylabel('Energy (eV)', fontsize=fontSz)
    plt.xlabel('k-path', fontsize=fontSz)
    plt.tick_params(labelsize=fontSz)
    # plt.subplot(122, title="path plot")
    # this_model.lattice.plot_brillouin_zone(decorate=False)
    # bands.plot_kpath()
    plt.show()
    plt.clf()


if __name__ == '__main__':
    time1 = time.time()
    # write here
    extend = 2
    # gen_bigger_lat(ex=extend, name='MoS2')
    ex_lat, lat_acc = gen_bigger_lat(ex=extend, name='MoS2')  # ex 为扩胞参数
    # ex_lat, lat_acc = initial_gen_method(ex=extend, name='MoS2')  # ex 为扩胞参数
    # print('extend =', extend)
    path_integral(lat=ex_lat, acc=lat_acc)

    # model = pb.Model(ex_lat,
    #                  pb.translational_symmetry())
    # model.lattice.plot()
    # plt.axis('off')
    # plt.title('extend %dX%d model' % (extend, extend))
    # # plt.axis('off')
    # plt.show()
    # plt.clf()
    #
    # model.lattice.plot_brillouin_zone()
    # plt.title('extend brillouin')
    # plt.show()
    # plt.clf()
    #
    # solver = pb.solver.lapack(model)
    #
    # k_points = model.lattice.brillouin_zone()
    # gamma = [0, 0]
    # k = k_points[0]
    # m = (k_points[0] + k_points[1]) / 2
    # # M_ = [0, 2 * np.pi / (3 * lat_acc)]
    # # K2_ = [2 * np.pi / (3 * 3 ** 0.5 * lat_acc), 2 * np.pi / (3 * lat_acc)]
    #
    # path_bands = []  # 重写path band方法
    #
    # this_path = pb.results.make_path(gamma, k, m, gamma, step=0.3)
    # # this_path = pb.results.make_path(gamma, K2_, M_, gamma, step=0.01)
    # # pd.DataFrame(model.hamiltonian.todense()).to_excel('data/model2hamiltonian_ex=1.xls', header=True, index=True,)
    # for kp in this_path:
    #     solver.set_wave_vector(kp)
    #     path_bands.append(solver.eigenvalues)
    #     # solver.calc_eigenvalues().plot_heatmap(show_indices=True)
    #     # print(kp, ' >> ', solver.eigenvalues)
    # # print('eigenvalues >> ', solver.eigenvalues)
    #
    # result = pb.results.Bands(this_path, path_bands)
    # plt.title("MoS2 3-band model band structure")
    # # result.plot(lw=0.5)
    # result.plot(point_labels=[r"$\Gamma$", "K", "M", r"$\Gamma$"])
    # # plt.plot(np.array(path_bands)[0, :])
    # plt.axhline(y=0, linestyle='--', color='grey')
    # # plt.ylim(-0.5, 0.5)
    # plt.show()
    # plt.clf()

    # ar_solver = pb.solver.arpack(model, k=20)  # for the 20 lowest energy eigenvalues
    #
    # eigenvalues = ar_solver.calc_eigenvalues(map_probability_at=[-0.5, 0.5])  # position in [nm]
    # eigenvalues.plot_heatmap(show_indices=True)
    # pb.pltutils.colorbar()
    # plt.show()
    # plt.clf()
    #
    # probability_map = ar_solver.calc_probability(3)
    # probability_map.plot()
    # plt.show()
    # plt.clf()
    #
    # dos = ar_solver.calc_dos(energies=np.linspace(-1, 1, 200), broadening=0.05)  # [eV]
    # dos.plot()
    # plt.show()
    # plt.clf()

    # plt.title("Band structure path in reciprocal space")
    # model.lattice.plot_brillouin_zone(decorate=False)
    # result.plot_kpath(point_labels=[r"$\Gamma$", "K", "M", r"$\Gamma$"])
    #
    # plt.show()
    # plt.clf()

    # kpm = pb.kpm(model)
    #
    # eg = 6
    # broadening = 0.1
    # dos = kpm.calc_dos(energy=np.linspace(-eg, eg, 500), broadening=broadening, num_random=100)
    # dos.plot()
    # plt.show()
    # plt.clf()
    #
    # ldos = kpm.calc_ldos(energy=np.linspace(-eg, eg, 500), broadening=broadening,
    #                      position=[0, 0])
    # ldos.plot()
    # plt.title('ldos %.2f' % broadening)
    # plt.show()
    # plt.clf()

    time2 = time.time()

    print('>> Finished, use time %d s' % (time2 - time1))
