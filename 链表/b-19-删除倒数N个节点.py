# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        if not head or not head.next:
            return None

        h = ListNode(next=head)
        p  = h
        q = r = head

        for _ in range(n-1):
            r = r.next
        while r.next:
            r = r.next
            q = q.next
            p = p.next

        p.next = q.next
        return h.next
