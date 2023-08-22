#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   1005_继续3n+1猜想.py    
@Time    :   2021/11/19 10:04  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   验证n的时候，验证序列出现过n则n为关键字
'''
import copy


def isodd(x):
    if x % 2 != 0:
        return True
    return False


def Callatz(n, cL):
    # print(int(n), end='  ')
    # print()
    if isodd(n):
        if n == 1:
            cL.append(1)
            return cL
        cL.append(int(n))
        return Callatz((3 * n + 1) / 2, cL=cL)
    else:
        cL.append(int(n))
        return Callatz(n / 2, cL=cL)


def findTrue(tD, val, tmD):
    if tD == {}:
        return tmD
    for kk, vv in tD.items():
        if set(vv).issubset(set(val)):
            tmD[kk] = True
            tD.pop(kk)
            return findTrue(tD, val, tmD)
        else:
            tmD[kk] = False
            tD.pop(kk)
            return findTrue(tD, val, tmD)


def findSubset(dN, key=None):
    ls = []
    if key is None:
        for kl, vl in dN.items():
            tmpDic = copy.deepcopy(dN)
            tmpDic.pop(kl)
            # print(k)
            tDic = {}
            bigTrue = findTrue(tmpDic, vl, tmD=tDic)
            yes = False
            for i in bigTrue.values():
                if ~i:
                    yes = True
                    break
            if yes:
                ls.append([kn for kn, vn in bigTrue.items() if vn == True])
    return ls


if __name__ == '__main__':
    k = int(input())
    nbL = [None for i in range(k)]
    numStr = sorted([int(i) for i in input().split()])
    dicNum = {}
    for i in range(k):
        dicNum[numStr[i]] = Callatz(numStr[i], cL=[])
    allLis = findSubset(dicNum)
    notInLis = []
    for k in dicNum:
        flag = True
        for j in allLis:
            if {k}.issubset(set(j)):
                flag = False
                break
        if flag:
            notInLis.append(k)

    for i in sorted(notInLis, reverse=True)[:len(notInLis)-1]:
        print(i, end=' ')
    print(sorted(notInLis, reverse=True)[-1])
