class Solution:
    def myPow(self, x: float, n: int) -> float:
        def f(x, n):
            if n == 0:
                return 1
            if n == 1:
                return x
            if n == -1:
                return 1/ x
            t = f(x, n//2)
            if abs(n) % 2 == 1:
                return t * t * x
            return t * t
        return f(x, n)