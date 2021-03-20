# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        # head-tail反转变成了tail-head且已经和下一段的头完成了连接
        def reverse(head, tail):
            pre = tail.next
            cur = head
            while pre != tail:
                nex = cur.next
                cur.next = pre
                pre, cur = cur, nex
        # 虚拟头结点
        pre = vh = ListNode(next=head)
        p = head
        while p:
            q = p
            for i in range(k-1):
                q = q.next
                if not q:
                    return vh.next
            # p-q反转后变成q-p
            reverse(p, q)
            pre.next = q
            pre, p = p, p.next
        return vh.next



