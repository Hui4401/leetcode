class Solution:
    def rob(self, nums: [int]) -> int:
        # l = len(nums)
        # if l == 0:
        #     return 0
        # nums[0] = (nums[0], 0)
        # for i in range(1, l):
        #     nums[i] = (nums[i-1][1] + nums[i], max(nums[i-1][0], nums[i-1][1]))
        # return max(nums[l-1][0], nums[l-1][1])
        
        a, b = 0, 0
        for num in nums:
            a, b = b + num, max(a, b)
        return max(a, b)
