# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        # 243, 564
        p, q = l1, l2
        root = ListNode()
        r = root
        j = 0
        while(p and q):
            r.next = ListNode((p.val + q.val + j) % 10)
            r = r.next
            j = (p.val + q.val + j) // 10
            p = p.next
            q = q.next
        if p:
            while p:
                r.next = ListNode((p.val + j) % 10)
                r = r.next
                j = (p.val + j) // 10
                p = p.next
        if q:
            while q:
                r.next = ListNode((q.val + j) % 10)
                r = r.next
                j = (q.val + j) // 10
                q = q.next
        if j:
            r.next = ListNode(j)
        return root.next