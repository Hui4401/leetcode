# # dp
# class Solution:
#     def cuttingRope(self, n: int) -> int:
#         dp = [0 for _ in range(n+1)]
#         dp[2] = 1
#         for i in range(3, n+1):
#             for j in range(1, i-1):
#                 dp[i] = max(dp[i], max(j*(i-j), dp[i-j]*j))
#         return dp[-1]

class Solution:
    def cuttingRope(self, n: int) -> int:
        if n <= 3:
            return n-1
        res = 1
        # 分出尽可能多的3，4比较特殊分为2*2更好，所以剩余<=4时就直接将剩下的相乘返回
        while n > 4:
            res *= 3
            n -= 3
        return res * n