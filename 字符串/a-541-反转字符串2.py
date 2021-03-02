class Solution:
    def reverseStr(self, s: str, k: int) -> str:
        n = len(s)
        g = n // (k * 2)
        r = n % (k * 2)
        res = ''
        for i in range(g):
            res += s[i*k*2:i*k*2+k][::-1]
            res += s[i*k*2+k:(i+1)*k*2]
        if r < k:
            res += s[g*k*2:][::-1]
        else :
            res += s[g*k*2:g*k*2+k][::-1]
            res += s[g*k*2+k:]
        return res