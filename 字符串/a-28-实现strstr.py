class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        if not needle:
            return 0

        def get_next(needle: str):
            nex = [-1 for _ in range(len(needle)+1)]
            i, j = 0, -1
            while i < len(needle):
                if j == -1 or needle[i] == needle[j]:
                    i += 1
                    j += 1
                    nex[i] = j
                else:
                    j = nex[j]
            return nex

        def kmp(haystack: str, needle: str, nex: [int]):
            i = j = 0
            while i < len(haystack) and j < len(needle):
                if j == -1 or haystack[i] == needle[j]:
                    i += 1
                    j += 1
                else:
                    j = nex[j]
            if j == len(needle):
                return i - j
            else:
                return -1

        nex = get_next(needle)
        print(nex)
        return kmp(haystack, needle, nex)

s = Solution()
s.strStr('mississippi', 'asdfasdfasdf')