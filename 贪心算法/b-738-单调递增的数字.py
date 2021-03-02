class Solution:
    def monotoneIncreasingDigits(self, N: int) -> int:
        N = list(str(N))
        n = len(N)
        # 标记开始变成9的最高位
        flag = n
        for i in range(n-1, 0, -1):
            if N[i-1] > N[i]:
                N[i-1] = chr(ord(N[i-1])-1)
                flag = i
        for i in range(flag, n):
            N[i] = '9'
        return int(''.join(N))