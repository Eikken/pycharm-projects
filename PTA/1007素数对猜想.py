#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   1007素数对猜想.py    
@Time    :   2021/12/13 14:55  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''


def isPrime(x, pL):
    for val in pL:
        if val > x**0.5:
            break
        if x % val == 0:
            return False
    # for val in range(2, x // 2):
    #     if x % val == 0:
    #         return False
    return True


if __name__ == '__main__':
    # n = int(input())
    # n = 20
    n = int(input())
    primeList = [2]
    thatPrime = 3
    count = 0
    for i in range(3, n + 1):
        if isPrime(i, primeList):
            primeList.append(i)
            thisPrime = thatPrime
            thatPrime = i
            if thisPrime - thatPrime == -2:
                count += 1
    print(count)
