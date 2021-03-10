"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""
# class Solution:
#     def copyRandomList(self, head: 'Node') -> 'Node':
#         if not head:
#             return None
#         d = {}
#         p = head
#         while p:
#             d[p] = Node(p.val)
#             p = p.next
#         p = head
#         while p:
#             d[p].next = d.get(p.next)
#             d[p].random = d.get(p.random)
#             p = p.next
#         return d[head]

class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head:
            return None
        # 构建拼接节点
        p = head
        while p:
            node = Node(p.val, next = p.next)
            p.next = node
            p = node.next
        # 构建各新节点的 random 指向
        p = head
        while p:
            if p.random:
                p.next.random = p.random.next
            p = p.next.next
        # 拆分节点
        pre = head
        cur = head.next
        p = cur
        # 注意末尾
        while cur.next:
            pre.next = pre.next.next
            cur.next = cur.next.next
            pre, cur = pre.next, cur.next
        pre.next = None
        return p