class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        sumn = sum(nums)
        if sumn % 2 == 1:
            return False
        target = sumn // 2
        dp = [0] * (target + 1)
        for i in range(len(nums)):
            j = target
            while j >= nums[i]:
                dp[j] = max(dp[j], dp[j-nums[i]]+nums[i])
                j -= 1
        if dp[target] == target:
            return True
        return False