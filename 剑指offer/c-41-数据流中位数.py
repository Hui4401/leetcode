import heapq
class MedianFinder:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.heapmax = []
        self.heapmin = []
        heapq.heapify(self.heapmax)
        heapq.heapify(self.heapmin)


    def addNum(self, num: int) -> None:
        heapq.heappush(self.heapmax, -num)
        heapq.heappush(self.heapmin, -heapq.heappop(self.heapmax))
        if len(self.heapmin) - len(self.heapmax) >= 1:
            heapq.heappush(self.heapmax, -heapq.heappop(self.heapmin))


    def findMedian(self) -> float:
        if len(self.heapmax) == len(self.heapmin):
            return (self.heapmin[0] - self.heapmax[0]) / 2
        return -self.heapmax[0]



# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()