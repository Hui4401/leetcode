class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        n = len(prices)
        if n == 1:
            return 0
        tmp = []
        for i in range(1, n):
            tmp.append(prices[i]-prices[i-1])
        res = 0
        h = 0
        for t in tmp:
            if t <= 0 and h == 0:
                continue
            h += t
            if h < 0:
                h = 0
            elif h > 2:
                res += h - 2
                h = 0
        return res
