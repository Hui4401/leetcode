class Solution:
    def reverseWords(self, s: str) -> str:
        s = s.strip()
        n = len(s)
        i, j = n-1, n
        res = ''
        while i:
            if s[i] == ' ':
                res += s[i+1:j]
                while s[i] == ' ':
                    i -= 1
                j = i + 1
            else:
                i -= 1
        res += s[:j]
        return res