class Solution:
    def rob1(self, nums: [int]) -> int:
        a, b = 0, 0
        for num in nums:
            a, b = b + num, max(a, b)
        return max(a, b)

    def rob(self, nums: [int]) -> int:
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        a = self.rob1(nums[1:])
        b = self.rob1(nums[: -1])
        return max(a, b)