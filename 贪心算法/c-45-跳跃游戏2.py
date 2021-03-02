class Solution:
    def jump(self, nums: List[int]) -> int:
        n = len(nums)
        res = 0
        cur_cover = next_cover = 0
        for i in range(n-1):
            next_cover = max(i + nums[i], next_cover)
            if i == cur_cover:
                cur_cover = next_cover
                res += 1
        return res