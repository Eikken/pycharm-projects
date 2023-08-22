#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   FizzBuzz.py    
@Time    :   2022/4/22 11:16  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''


class Solution(object):
    def fizzBuzz(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        nArr = []
        for i in range(1, n + 1):
            if i % 3 == 0 and i % 5 == 0:
                nArr.append('FizzBuzz')
            elif i % 3 == 0 and i % 5 != 0:
                nArr.append('Fizz')
            elif i % 3 != 0 and i % 5 == 0:
                nArr.append('Buzz')
            else:
                nArr.append(str(i))
        return nArr


if __name__ == '__main__':
    n = int(input())
    sol = Solution()
    print(sol.fizzBuzz(n))
