class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        res = 0
        for i in range(1, len(prices)):
            subn = prices[i] - prices[i-1]
            if subn > 0:
                res += subn
        return res