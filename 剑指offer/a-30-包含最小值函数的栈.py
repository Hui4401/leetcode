class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []


    def push(self, x: int) -> None:
        if not self.stack:
            # 第二个数表示当前栈中的最小值
            self.stack.append((x, x))
        else:
            _, minn = self.stack[-1]
            if x < minn:
                self.stack.append((x, x))
            else:
                self.stack.append((x, minn))


    def pop(self) -> None:
        self.stack.pop(-1)

    def top(self) -> int:
        x, _ = self.stack[-1]
        return x

    def min(self) -> int:
        _, minn = self.stack[-1]
        return minn



# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.min()