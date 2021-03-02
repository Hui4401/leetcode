# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        if not head:
            return None
        fast = low = head
        while True:
            if not fast.next or not fast.next.next:
                return None
            fast = fast.next.next
            low = low.next
            if fast == low:
                break
        fast = head
        while True:
            if fast == low:
                return low
            fast = fast.next
            low = low.next