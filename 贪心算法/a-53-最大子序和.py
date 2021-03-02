class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        if not nums:
            return 0
        res = -2**31
        count = 0
        for num in nums:
            count += num
            if count > res:
                res = count
            if count < 0:
                count = 0
        return res