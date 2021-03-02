class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n = len(nums)
        if n < 2:
            return True
        i = cover = 0
        while i <= cover:
            # i + nums[i] 为从位置i可以跳到的最远位置
            cover = max(i + nums[i], cover)
            if cover >= n -1:
                return True
        return False
