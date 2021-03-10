class Solution:
    def findNthDigit(self, n: int) -> int:
        digit = 1
        start = 1
        count = 9
        while n > count:
            n -= count
            digit += 1
            start *= 10
            count = digit * start * 9
        num = start + (n-1) // digit
        return int(str(num)[(n-1)%digit])