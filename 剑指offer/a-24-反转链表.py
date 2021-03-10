# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        h = ListNode(0)
        h.next = None
        p = head
        while p:
            q = p.next
            p.next = h.next
            h.next = p
            p = q
        return h.next