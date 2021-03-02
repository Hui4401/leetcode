class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        g.sort(reverse=True)
        s.sort(reverse=True)
        res = 0
        i = j = 0
        ng, ns = len(g), len(s)
        while i < ng and j < ns:
            if s[j] >= g[i]:
                res += 1
                j += 1
            i += 1
        return res
