"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""

class Solution:
    def postorder(self, root: 'Node') -> List[int]:
        if not root:
            return []
        res = []
        stack = [root]
        while stack:
            node = stack.pop(-1)
            if node:
                stack.append(node)
                stack.append(None)
                for i in node.children[::-1]:
                    stack.append(i)
            else:
                node = stack.pop(-1)
                res.append(node.val)
        return res