from collections import deque


class MyQueue:
    def __init__(self):
        # 不用list是因为pop(0)时开销太大
        self.queue = deque()
    def pop(self, value):
        if self.queue and value == self.queue[0]:
            # pop和append默认都是在右边，操作左边加上left
            self.queue.popleft()
    def push(self, value):
        while self.queue and value > self.queue[-1]:
            self.queue.pop()
        self.queue.append(value)
    def get(self):
        return self.queue[0]


class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        queue = MyQueue()
        res = []
        # 先将前k的元素放进队列
        for i in range(k):
            queue.push(nums[i])
        res.append(queue.get())
        # i代表窗口的尾部索引
        for i in range(k, len(nums)):
            queue.pop(nums[i-k])
            queue.push(nums[i])
            res.append(queue.get())
        return res