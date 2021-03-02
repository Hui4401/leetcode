from collections import Counter


class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        cr = dict(Counter(ransomNote))
        cm = dict(Counter(magazine))
        for key, value in cr.items():
            if key not in cm.keys():
                return False
            if value > cm[key]:
                return False
        return True