class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        for c in s:
            if c in ('(', '{', '['):
                stack.append(c)
            else:
                if not stack:
                    return False
                t = stack.pop(-1)
                if c == ')' and t != '(' or c == '}' and t != '{' or c == ']' and t != '[':
                    return False
        return False if stack else True