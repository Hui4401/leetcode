class MyStack:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.queue1 = []
        self.queue2 = []


    def push(self, x: int) -> None:
        """
        Push element x onto stack.
        """
        self.queue1.append(x)


    def pop(self) -> int:
        """
        Removes the element on top of the stack and returns that element.
        """
        x = self.queue1.pop(0)
        while self.queue1:
            self.queue2.append(x)
            x = self.queue1.pop(0)
        while self.queue2:
            self.queue1.append(self.queue2.pop(0))
        return x



    def top(self) -> int:
        """
        Get the top element.
        """
        x = self.queue1.pop(0)
        while self.queue1:
            self.queue2.append(x)
            x = self.queue1.pop(0)
        while self.queue2:
            self.queue1.append(self.queue2.pop(0))
        self.queue1.append(x)
        return x


    def empty(self) -> bool:
        """
        Returns whether the stack is empty.
        """
        return False if self.queue1 else True



# Your MyStack object will be instantiated and called as such:
# obj = MyStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.empty()