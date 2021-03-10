class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)
        # 0表示空串，数组要长一位
        dp = [[False for _ in range(n+1)] for _ in range(m+1)]

        # 初始化
        dp[0][0] = True
        for i in range(2, n+1, 2):
            if dp[0][i-2] and p[i-1] == '*':
                dp[0][i] = True

        for i in range(1, m+1):
            for j in range(1, n+1):
                # p[j-1] 不为 * 只有两种情况下可以匹配
                if p[j-1] != '*':
                    if dp[i-1][j-1] and p[j-1] == s[i-1]:
                        dp[i][j] = True
                    elif dp[i-1][j-1] and p[j-1] == '.':
                        dp[i][j] = True
                # p[j-1] 为 *，则有三种情况
                else:
                    if dp[i][j-2]:
                        dp[i][j] = True
                    elif dp[i-1][j] and p[j-2] == s[i-1]:
                        dp[i][j] = True
                    elif dp[i-1][j] and p[j-2] == '.':
                        dp[i][j] = True
        return dp[-1][-1]