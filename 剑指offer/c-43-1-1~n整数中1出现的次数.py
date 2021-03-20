class Solution:
    def countDigitOne(self, n: int) -> int:
        digit = 1
        high, cur, low = n // 10, n % 10, 0
        res = 0
        while high != 0 or cur != 0:
            res += high * digit
            if cur == 1:
                res += low + 1
            elif cur != 0:
                res += digit
            low += cur * digit
            cur = high % 10
            high //= 10
            digit *= 10
        return res