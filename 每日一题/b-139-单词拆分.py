class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        st = set(wordDict)
        dp = [False] * (len(s) + 1)
        dp[0] = True
        for i in range(len(s)):
            for j in range(i+1, len(s)+1):
                if dp[i] and s[i:j] in st:
                    dp[j] = True
        return dp[-1]