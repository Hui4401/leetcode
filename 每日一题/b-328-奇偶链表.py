# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def oddEvenList(self, head: ListNode) -> ListNode:
        h1 = ListNode()
        h2 = ListNode()
        p1 = h1
        p2 = h2

        index = 1
        while head:
            if index % 2 == 1:
                p1.next = head
                p1 = p1.next
            else:
                p2.next = head
                p2 = p2.next
            index += 1
            head = head.next

        p2.next = None
        p1.next = h2.next
        return h1.next
