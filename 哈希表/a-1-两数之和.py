class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        n = len(nums)
        if n < 2:
            return []
        d = dict()
        for i in range(n):
            c = target - nums[i]
            if c in d.keys():
                return [i, d[c]]
            if nums[i] not in d.keys():
                d[nums[i]] = i
        return []
