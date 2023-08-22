#!/usr/bin/env python

from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import colorConverter
from ase.units import Bohr
import os.path
from collections import namedtuple


def get_segment_midpoint(segments_):
    seg_list = []
    for seg in segments_:
        x_ = sum(seg[:, 0]) / 2
        y_ = sum(seg[:, 1]) / 2
        seg_list.append([x_, y_])

    return np.array(seg_list)


def plot_band_weight(kslist,
                     ekslist,
                     wkslist=None,
                     efermi=0,
                     yrange=None,
                     output=None,
                     style='alpha',
                     color='blue',
                     axis=None,
                     width=2,
                     xticks=None,
                     title=None):
    if axis is None:
        fig, a = plt.subplots()
        plt.tight_layout(pad=2.19)
        plt.axis('tight')
        plt.gcf().subplots_adjust(left=0.17)
    else:
        a = axis
    if title is not None:
        a.set_title(title)

    xmax = max(kslist[0])
    if yrange is None:
        yrange = (np.array(ekslist).flatten().min() - 66,
                  np.array(ekslist).flatten().max() + 66)

    # if
    file_name = "this_xyz.csv"
    xy = []

    if wkslist is not None:

        for i in range(len(kslist)):
            x = kslist[i]
            y = ekslist[i]
            lwidths = np.array(wkslist[i]) * width
            # lwidths=np.ones(len(x))
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            rgba_list = [colorConverter.to_rgba(color, alpha=np.abs(lwidth / (width + 0.001))) for lwidth in lwidths]
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            # seg_arr = get_segment_midpoint(segments)
            alpha_c = [c[3] for c in rgba_list]
            if i == 0:
                # 添加一列mid point [x, y, a], 透明度维度是 len(x)-1, y 维度是 len(x) ??? 透明度维度是 len(x) ???
                xy = np.stack([x, y, alpha_c], axis=1)  # x y1 a1
            else:
                temp = np.stack([x, y, alpha_c], axis=1)
                xy = np.concatenate((xy, temp), axis=0)
            if style == 'width':
                lc = LineCollection(segments, linewidths=lwidths, colors=color)
            elif style == 'alpha':
                lc = LineCollection(
                    segments,
                    linewidths=[2] * len(x),
                    colors=rgba_list)
            a.add_collection(lc)
    if os.path.exists(file_name):  #
        os.remove(file_name)  #
    import pandas as pd
    pd.DataFrame(xy).to_csv(file_name)
    # xy 存储为： [x y1 z1 y2 z2 y3 z3] ...  x1 y1 是 segment line 的中点值，z反映了rgba的透明度。
    #            [0  1  2  3  4  5  6]
    # endif
    plt.ylabel('Frequency (cm$^{-1}$)')
    if axis is None:
        for ks, eks in zip(kslist, ekslist):
            plt.plot(ks, eks, color='gray', linewidth=0.001)
        a.set_xlim(0, xmax)
        a.set_ylim(yrange)
        if xticks is not None:
            plt.xticks(xticks[1], xticks[0])
        for x in xticks[1]:
            plt.axvline(x, color='gray', linewidth=0.5)
        if efermi is not None:
            plt.axhline(linestyle='--', color='black')
    return a
