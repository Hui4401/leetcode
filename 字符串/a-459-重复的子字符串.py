class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        def get_next(needle: str):
            nex = [-1 for _ in range(len(needle)+1)]	# 长度多一位，保留右移前最后一个next，某些情况下有用
            i, j = 0, -1
            while i < len(needle):
                if j == -1 or needle[i] == needle[j]:
                    i += 1
                    j += 1
                    nex[i] = j
                else:
                    j = nex[j]
            return nex
        n = len(s)
        if n < 2:
            return False
        nex = get_next(s)
        if nex[n] != 0 and  n % (n - nex[n]) == 0:
            return True
        return False
