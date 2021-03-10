from collections import deque
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        if not nums:
            return []
        queue = deque()
        for i in range(k):
            if not queue:
                queue.append(nums[i])
            else:
                while queue and queue[-1] < nums[i]:
                    queue.pop()
                queue.append(nums[i])
        res = [queue[0]]
        for i in range(1, len(nums)-k+1):
            if nums[i-1] == queue[0]:
                queue.popleft()
            while queue and queue[-1] < nums[i+k-1]:
                queue.pop()
            queue.append(nums[i+k-1])
            res.append(queue[0])
        return res

