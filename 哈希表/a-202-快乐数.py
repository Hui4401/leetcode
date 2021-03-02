class Solution:
    def isHappy(self, n: int) -> bool:
        def summ(n: int):
            s = 0
            while n:
                x = n % 10
                n = n // 10
                s += x * x
            return s
        s = {n}
        while n != 1:
            n = summ(n)
            if n in s:
                return False
            s.add(n)
        return True
