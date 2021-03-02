# class Solution:
#     def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
#         n = len(gas)
#         for i in range(n):
#             have = gas[i] - cost[i]
#             j = (i + 1) % n
#             while have > 0 and j != i:
#                 have += gas[j] - cost[j]
#                 j  = (j + 1) % n
#             if have >= 0 and j == i:
#                 return i
#         return -1

class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        # 当前剩余油量和
        cur = 0
        # 总剩余油量和
        total = 0
        # 可能的出发位置
        start = 0
        for i in range(len(gas)):
            cur += gas[i] - cost[i]
            total += gas[i] - cost[i]
            if cur < 0:
                start = i + 1
                cur = 0
        if total >= 0:
            return start
        return -1