from itertools import permutations
from typing import List


class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        return list(set(list(permutations(nums))))


s = Solution()
r = s.permuteUnique([1, 1, 2])
print(r)
