# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
from collections import deque
class Solution:
    def findBottomLeftValue(self, root: TreeNode) -> int:
        if not root:
            return None
        queue = deque([root])
        res = 0
        while queue:
            size = len(queue)
            for _ in range(size):
                node = queue.popleft()
                res = node.val
                if node.right:
                    queue.append(node.right)
                if node.left:
                    queue.append(node.left)
        return res