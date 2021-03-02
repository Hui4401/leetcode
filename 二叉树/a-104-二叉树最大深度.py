# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        res = 0
        def dfs(root, depth):
            if not root:
                return
            nonlocal res
            if depth > res:
                res = depth
            dfs(root.left, depth+1)
            dfs(root.right, depth+1)
        dfs(root, 1)
        return res

# class Solution:
#     def maxDepth(self, root: TreeNode) -> int:
#         def dfs(root):
#             if not root:
#                 return 0
#             d1 = dfs(root.left)
#             d2 = dfs(root.right)
#             return max(d1, d2) + 1
#         return dfs(root)

# from collections import deque
# class Solution:
#     def maxDepth(self, root: TreeNode) -> int:
#         if not root:
#             return 0
#         queue = deque([root])
#         res = 0
#         while queue:
#             size = len(queue)
#             for i in range(size):
#                 node = queue.popleft()
#                 if node.left:
#                     queue.append(node.left)
#                 if node.right:
#                     queue.append(node.right)
#             res += 1
#         return res