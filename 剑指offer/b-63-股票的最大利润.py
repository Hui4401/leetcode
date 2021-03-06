class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) <= 1:
            return 0
        res = tmp = 0
        for i in range(1, len(prices)):
            tmp += prices[i] - prices[i-1]
            if tmp < 0:
                tmp = 0
            elif tmp > res:
                res = tmp
        return res