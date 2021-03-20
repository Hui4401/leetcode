class Solution:
    def isPalindrome(self, s: str) -> bool:
        if not s:
            return False
        s = s.lower()
        i, j = 0, len(s)-1
        while i < j:
            while i < len(s) and not ('0' <= s[i] <= '9' or 'a' <= s[i] <= 'z'):
                i += 1
            while j > -1 and not ('0' <= s[j] <= '9' or 'a' <= s[j] <= 'z'):
                j -= 1
            if i < j and s[i] != s[j]:
                return False
            i += 1
            j -= 1
        return True