class Solution:
    def climbStairs(self, n: int) -> int:
        if n < 3:
            return n
        dp1, dp2 = 1, 2
        for _ in range(3, n+1):
            dp1, dp2 = dp2, dp1 + dp2
        return dp2