from collections import Counter
from queue import PriorityQueue
import heapq


# class Solution:
#     def topKFrequent(self, nums: List[int], k: int) -> List[int]:
#         counter = dict(Counter(nums))
#         heap = []
#         for key, value in counter.items():
#             # 由于heapq默认实现小顶堆，对权值取反来得到大顶堆
#             heap.append((-value, key))
#         # 将list转换为堆结构
#         heapq.heapify(heap)
#         res = []
#         while k:
#             res.append(heapq.heappop(heap)[1])
#             k -= 1
#         return res


class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        counter = dict(Counter(nums))
        pq = PriorityQueue()
        for key, value in counter.items():
            # PriorityQueue也是实现小顶堆，所以对权值取反来得到大顶堆
            pq.put((-value, key))
        res = []
        while k:
            res.append(pq.get()[1])
            k -= 1
        return res