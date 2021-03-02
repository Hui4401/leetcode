# class Solution:
#     def fib(self, n: int) -> int:
#         if n == 0:
#             return 0
#         if n == 1:
#             return 1
#         return self.fib(n-1) + self.fib(n-2)

# class Solution:
#     def fib(self, n: int) -> int:
#         if n < 2:
#             return n
#         dp = [0 for _ in range(n+1)]
#         dp[0] = 0
#         dp[1] = 1
#         for i in range(2, n+1):
#             dp[i] = dp[i-1] + dp[i-2]
#         return dp[-1]

class Solution:
    def fib(self, n: int) -> int:
        if n < 2:
            return n
        l, r = 0, 1
        for _ in range(2, n+1):
            l, r = r, l + r
        return r