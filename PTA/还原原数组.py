#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   还原原数组.py    
@Time    :   2022/4/22 11:24  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''


class Solution(object):

    def recoverArray(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        halfLenNums = int(len(nums) / 2)
        newList = set([i + j for i in nums for j in nums[1:]])
        print(halfLenNums)
        # print(newList)
        if halfLenNums == 1:
            newList = list(newList)
            newList.sort()
            nArr = list(map(int, [newList[0] / 2]))
            return nArr
        for halfNum in newList:
            lower = [i for i in nums if i <= halfNum]
            higher = [i for i in nums if i >= halfNum]
            print(lower, higher)
            # if len(lower) or len(higher) == halfLen(nums) +1 or -1 (valid）
            if len(lower) == halfLenNums:
                higher = [i for i in nums if i not in lower]
                higher.sort()
                lower.sort()
                nArr = [int((i + j) / 2) for i, j in zip(lower, higher)]
                print(nArr)

            if len(higher) == halfLenNums:
                lower = [i for i in nums if i not in higher]
                higher.sort()
                lower.sort()
                nArr = [int((i + j) / 2) for i, j in zip(lower, higher)]
                print(nArr)


if __name__ == '__main__':
    sol = Solution()
    n = [11, 6, 3, 4, 8, 7, 8, 7, 9, 8, 9, 10, 10, 2, 1, 9]
    sol.recoverArray(n)
