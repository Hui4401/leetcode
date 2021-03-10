class Solution:
    def add(self, a: int, b: int) -> int:
        x = 0xffffffff
        a, b = a & x, b & x
        while b != 0:
            c = (a & b) << 1 & x
            a = a ^ b
            b = c
        return ~(a ^ x) if a & (1<<31) else a
