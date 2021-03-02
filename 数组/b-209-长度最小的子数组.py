from typing import List


class Solution:
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        res = 2**31 - 1
        i = 0
        su = 0
        for j in range(len(nums)):
            su += nums[j]
            while su >= s:
                l = j - i + 1
                if l < res:
                    res = l
                su -= nums[i]
                i += 1
        return 0 if res == 2**31-1 else res


s = Solution()
a = s.minSubArrayLen(7, [2,3,1,2,4,3])
print(a)

