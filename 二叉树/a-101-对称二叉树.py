# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# class Solution:
#     def isSymmetric(self, root: TreeNode) -> bool:
#         if not root:
#             # 空树居然是对称的
#             return True
#         # 同时遍历两颗子树
#         def compare(left, right):
#             # 一个为空一个不为空
#             if left and not right or right and not left:
#                 return False
#             # 两个都为空
#             if not left and not right:
#                 return True
#             # 数值不相等
#             if left.val != right.val:
#                 return False
#             # 本层对称，比较孩子（左边的左孩子和右边的右孩子比较）
#             c1 = compare(left.left, right.right)
#             c2 = compare(left.right, right.left)
#             return True if c1 and c2 else False
#         return compare(root.left, root.right)

from collections import deque
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        if not root:
            return True
        queue = deque([root.left, root.right])
        while queue:
            node1 = queue.popleft()
            node2 = queue.popleft()
            if not node1 and not node2:
                continue
            if not node1 or not node2 or node1.val != node2.val:
                return False
            queue.append(node1.left)
            queue.append(node2.right)
            queue.append(node1.right)
            queue.append(node2.left)
        return True
