# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# # 递归
# class Solution:
#     def invertTree(self, root: TreeNode) -> TreeNode:
#         def dfs(root):
#             if not root:
#                 return
#             root.left, root.right = root.right, root.left
#             dfs(root.left)
#             dfs(root.right)
#         dfs(root)
#         return root

# # 迭代统一写法
# class Solution:
#     def invertTree(self, root: TreeNode) -> TreeNode:
#         if not root:
#             return None
#         stack = [root]
#         while stack:
#             node = stack.pop(-1)
#             if node:
#                 if node.right:
#                     stack.append(node.right)
#                 if node.left:
#                     stack.append(node.left)
#                 stack.append(node)
#                 stack.append(None)
#             else:
#                 node = stack.pop(-1)
#                 node.left, node.right = node.right, node.left
#         return root

# 层序
from collections import deque
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return None
        queue = deque([root])
        while queue:
            node = queue.popleft()
            node.left, node.right = node.right, node.left
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        return root