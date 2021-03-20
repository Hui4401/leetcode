class Solution:
    def translateNum(self, num: int) -> int:
        if num < 10:
            return 1
        strs = str(num)
        n = len(strs)
        dp = [1 for _ in range(n)]
        dp[1] = 1 if strs[:2] > "25" else 2
        for i in range(2, n):
            if strs[i-1:i+1] > "25" or strs[i-1:i+2] < "10":
                dp[i] = dp[i-1]
            else:
                dp[i] = dp[i-1] + dp[i-2]
        return dp[-1]