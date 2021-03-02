"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""
from collections import deque
class Solution:
    def maxDepth(self, root: 'Node') -> int:
        if not root:
            return 0
        queue = deque([root])
        res = 0
        while queue:
            size = len(queue)
            for i in range(size):
                node = queue.popleft()
                for child in node.children:
                    queue.append(child)
            res += 1
        return res