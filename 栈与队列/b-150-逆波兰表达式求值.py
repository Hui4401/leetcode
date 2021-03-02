class Solution:
    def evalRPN(self, tokens: [str]) -> int:
        stack = []
        for i in tokens:
            if i in ('+', '-', '*', '/'):
                x2 = stack.pop(-1)
                x1 = stack.pop(-1)
                if i == '+':
                    stack.append(x1 + x2)
                elif i == '-':
                    stack.append(x1 - x2)
                elif i == '*':
                    stack.append(x1 * x2)
                else:
                    stack.append(int(x1 / x2))
            else:
                stack.append(int(i))
        return stack[-1]