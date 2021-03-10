class Solution:
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        stack = []
        m, n = len(pushed), len(popped)
        if m != n:
            return False
        i = j = 0
        while i < m and j < n:
            if pushed[i] != popped[j]:
                stack.append(pushed[i])
                i += 1
            else:
                i += 1
                j += 1
                while stack and stack[-1] == popped[j]:
                    stack.pop(-1)
                    j += 1
        if i < m or j < n:
            return False
        return True