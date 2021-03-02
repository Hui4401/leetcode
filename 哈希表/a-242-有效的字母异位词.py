from collections import Counter


class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        d1 = dict(Counter(s))
        d2 = dict(Counter(t))
        for key, value in d2.items():
            v = d1.get(key)
            if not v or v != value:
                return False
        return True