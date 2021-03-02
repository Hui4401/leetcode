class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        n1 = len(word1)
        n2 = len(word2)
        if not n1:
            return word2
        if not n2:
            return word1
        i = j = 0
        res = ''
        while i < n1 and j < n2:
            res += word1[i]
            res += word2[j]
            i += 1
            j += 1
        if i < n1:
            res += word1[i:]
        if j < n2:
            res += word1[j:]
        return res