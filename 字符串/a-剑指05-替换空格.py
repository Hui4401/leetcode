class Solution:
    def replaceSpace(self, s: str) -> str:
        res = ''
        for c in s:
            if c != ' ':
                res += c
            else:
                res += '%20'
        return res