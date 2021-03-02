# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeElements(self, head: ListNode, val: int) -> ListNode:
        h = ListNode(next=head)
        p = h
        while p.next:
            q = p.next
            if q.val == val:
                p.next = q.next
            else:
                p = p.next
        return h.next