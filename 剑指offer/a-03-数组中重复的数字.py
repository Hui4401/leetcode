from collections import defaultdict
class Solution:
    def findRepeatNumber(self, nums: List[int]) -> int:
        d = defaultdict(int)
        for num in nums:
            if d[num]:
                return num
            else:
                d[num] += 1