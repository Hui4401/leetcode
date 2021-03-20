class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        flag = [False for _ in range(128)]
        i = j = tmp = res = 0
        while j < len(s) and (len(s) - i) >= res:
            while flag[ord(s[j])]:
                flag[ord(s[i])] = False
                tmp -= 1
                i += 1
            flag[ord(s[j])] = True
            tmp += 1
            j += 1
            if tmp > res:
                res = tmp
        return res
