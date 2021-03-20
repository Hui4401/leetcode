class Solution:
    def dicesProbability(self, n: int) -> List[float]:
        dp = [[0 for _ in range(67)] for _ in range(12)]
        for i in range(1, 7):
            dp[1][i] = 1
        for i in range(2, n+1):
            for j in range(i, 6*i+1):
                for k in range(1, 7):
                    if j - k <= 0:
                        break
                    dp[i][j] += dp[i-1][j-k]
        total = 6 ** n
        res = []
        for i in range(n, n*6+1):
            res.append(dp[n][i]/total)
        return res