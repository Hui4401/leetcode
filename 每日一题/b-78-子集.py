from typing import List


class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        # [1, 2, 3]
        ret = []
        for num in nums:
            tmp = []
            for r in ret:
                tmp.append(r+[num])
            ret.extend(tmp)
            ret.append([num])
        ret.append([])
        return ret


s = Solution()
print(s.subsets([1, 2, 3]))