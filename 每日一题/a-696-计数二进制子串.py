class Solution:
    def countBinarySubstrings(self, s: str) -> int:
        l = len(s)
        if l == 1:
            return 0
        
        count = []
        now = s[0]
        nowc = 1
        for i in range(1, l):
            if s[i] == now:
                nowc += 1
            else:
                count.append(nowc)
                nowc = 1
                now = s[i]
        count.append(nowc)
        lc = len(count)
        if lc == 1:
            return 0
        
        ret = 0
        for i in range(1, lc):
            ret += min(count[i-1], count[i])

        return ret
