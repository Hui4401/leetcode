# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
from collections import deque

class Solution:
    def averageOfLevels(self, root: TreeNode) -> List[float]:
        if not root:
            return []
        res = []
        queue = deque()
        queue.append(root)
        while queue:
            size = len(queue)
            sumn = 0
            for _ in range(size):
                node = queue.popleft()
                sumn += node.val
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(sumn/size)
        return res