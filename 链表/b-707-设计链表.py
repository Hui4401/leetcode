class Node:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class MyLinkedList:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.h = Node(next=None)
        self.count = 0

    def get(self, index: int) -> int:
        """
        Get the value of the index-th node in the linked list. If the index is invalid, return -1.
        """
        if index >= self.count or index < 0:
            return -1
        p = self.h.next
        while index:
            p = p.next
            index -= 1
        return p.val

    def addAtHead(self, val: int) -> None:
        """
        Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list.
        """
        p = Node(val, next=self.h.next)
        self.h.next = p
        self.count += 1

    def addAtTail(self, val: int) -> None:
        """
        Append a node of value val to the last element of the linked list.
        """
        p = self.h
        while p.next:
            p = p.next
        p.next = Node(val, None)
        self.count += 1

    def addAtIndex(self, index: int, val: int) -> None:
        """
        Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted.
        """
        if index > self.count:
            return
        if index == self.count:
            self.addAtTail(val)
        elif index <= 0:
            self.addAtHead(val)
        else:
            p = self.h
            while index:
                p = p.next
                index -= 1
            q = Node(val, p.next)
            p.next = q
            self.count += 1

    def deleteAtIndex(self, index: int) -> None:
        """
        Delete the index-th node in the linked list, if the index is valid.
        """
        if index >= self.count or index < 0:
            return
        p = self.h
        while index:
            p = p.next
            index -= 1
        p.next = p.next.next
        self.count -= 1


# Your MyLinkedList object will be instantiated and called as such:
# obj = MyLinkedList()
# param_1 = obj.get(index)
# obj.addAtHead(val)
# obj.addAtTail(val)
# obj.addAtIndex(index,val)
# obj.deleteAtIndex(index)