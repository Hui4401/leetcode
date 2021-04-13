class Solution:
    def longestValidParentheses(self, s: str) -> int:
        stack = [-1]
        res = 0
        for i in range(len(s)):
            if s[i] == '(':
                stack.append(i)
            else:
                stack.pop(-1)
                if not stack:
                    stack.append(i)
                res = max(res, i-stack[-1])
        return res