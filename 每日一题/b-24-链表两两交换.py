# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head

        h = ListNode(next=head)
        p = h
        q = h.next
        r = q.next
        while True:
            q.next = r.next
            r.next = q
            p.next = r
            if not q.next or not q.next.next:
                break
            p = q
            q = q.next
            r = q.next
        return h.next