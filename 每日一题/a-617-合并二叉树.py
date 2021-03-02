# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def mergeTrees(self, t1: TreeNode, t2: TreeNode) -> TreeNode:
        def dfs(t1: TreeNode, t2: TreeNode):
            if not t2:
                return
            t1.val += t2.val
            if not t1.left and t2.left:
                t1.left = t2.left
                t2.left = None
            dfs(t1.left, t2.left)
            if not t1.right and t2.right:
                t1.right = t2.right
                t2.right = None
            dfs(t1.right, t2.right)

        if not t1 and not t2:
            return None
        if not t1:
            t1, t2 = t2, t1
        dfs(t1, t2)
        return t1