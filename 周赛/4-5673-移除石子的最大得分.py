class Solution:
    def maximumScore(self, a: int, b: int, c: int) -> int:
        minn = min(a, b, c)
        if minn == a:
            if b < c:
                midn, maxn = b, c
            else:
                midn, maxn = c, b
        elif minn == b:
            if a < c:
                midn, maxn = a, c
            else:
                midn, maxn = c, a
        else:
            if a < b:
                midn, maxn = a, b
            else:
                midn, maxn = b, a
        res = minn
        while minn:
            maxn -= 1
            if midn > maxn:
                midn, maxn = maxn, midn
            minn -= 1
        res += min(midn, maxn)
        return res