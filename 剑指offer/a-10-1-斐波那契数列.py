class Solution:
    def fib(self, n: int) -> int:
        if n < 2:
            return n
        pre, cur = 0, 1
        for _ in range(2, n+1):
            pre, cur = cur, pre + cur
        return cur % int(1e9+7)