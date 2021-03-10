class Solution:
    def lastRemaining(self, n: int, m: int) -> int:
        index = 0
        for i in range(2, n+1):
            index = (index + m) % i
        return index