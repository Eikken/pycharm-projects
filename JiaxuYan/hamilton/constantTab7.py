#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   constantTab7.py    
@Time    :   2023/4/3 10:15  
@E-mail  :   iamwxyoung@qq.com
@Tips    :
'''


# table MoS2 has checked
def Table7(MX2='MoS2'):
    name_ = MX2
    eps_dic = {
        # ->       eps1,   eps2,    eps3,    eps4,    eps5,    eps6,   eps7,   eps8,    eps9,    epsA,   epsB,
        'MoS2': [1.0688, 1.0688, -0.7755, -1.2902, -1.2902, -0.1380, 0.0874, 0.0874, -2.8949, -1.9065, -1.9065],
        'MoSe2': [],

    }
    # onsite refer sphere (n=1) Tij (i==j)
    onsite_dic = {
        # ->      t(1)11, t(1)22,  t(1)33, t(1)44,  t(1)55,  t(1)66, t(1)77,  t(1)88,  t(1)99, t(1)AA, t(1)BB (十六进制)
        'MoS2': [-0.2069, 0.0323, -0.1739, 0.8651, -0.1872, -0.2979, 0.2747, -0.5581, -0.1916, 0.9122, 0.0059],
        'MoSe2': [],

    }
    # offsite refer sphere (n=1) Tij (i!=j)
    offsite_dic = {
        # ->      t(1)35, t(1)68, t(1)9B,  t(1)12,  t(1)34,  t(1)45,  t(1)67,  t(1)78, t(1)9A, t(1)AB (十六进制)
        'MoS2': [-0.0679, 0.4096, 0.0075, -0.2562, -0.0995, -0.0705, -0.1145, -0.2487, 0.1063, -0.0385],
        'MoSe2': [],

    }
    # offsite refer sphere (n=5,6) Tij (i!=j)
    other_dic = {
        # ->      t(5)41, t(5)32,  t(5)52, t(5)96,  t(5)B6,  t(5)A7, t(5)98,  t(5)B8,  t(6)96,  t(6)B6, t(6)98, t(6)B8
        'MoS2': [-0.7883, -1.3790, 2.1584, -0.8836, -0.9402, 1.4114, -0.9535, 0.6517, -0.0686, -0.1498, -0.2205,
                 -0.2451],
        'MoSe2': [],

    }
    # 返回值：[0:11]是eps能量，[11:22]是eps能量，[22:32]是eps能量，[32:]是eps能量，
    return eps_dic[name_] + onsite_dic[name_] + offsite_dic[name_] + other_dic[name_]
