from collections import defaultdict


class Solution:
    def fourSumCount(self, A: List[int], B: List[int], C: List[int], D: List[int]) -> int:
        d = defaultdict(int)
        for i in A:
            for j in B:
                d[i+j] += 1
        count = 0
        for i in C:
            for j in D:
                if -i-j in d.keys():
                    count += d[-i-j]
        return count