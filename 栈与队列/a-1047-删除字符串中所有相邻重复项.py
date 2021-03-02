class Solution:
    def removeDuplicates(self, S: str) -> str:
        stack = []
        for c in S:
            if not stack or c != stack[-1]:
                stack.append(c)
            else:
                stack.pop(-1)
        return ''.join(stack)