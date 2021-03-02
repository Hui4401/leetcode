class Solution:
    def reverse(self, x: int) -> int:
        flag = True if x < 0 else False
        ret = int(str(abs(x))[::-1])
        if ret > 2**31-1:
            return 0
        if flag:
            ret = -ret
        return ret


s = Solution()
print(s.reverse(1563847412))