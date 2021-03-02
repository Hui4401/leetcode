# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def dfs(self, root: TreeNode) -> (int, int):
        if not root:
            return (0, 0)
        
        left = self.dfs(root.left)
        right = self.dfs(root.right)
        a = left[1] + right[1] + root.val
        b = max(left[0], left[1]) + max(right[0], right[1])
        
        return (a, b)

    def rob(self, root: TreeNode) -> int:
        ret = self.dfs(root)
        return max(ret[0], ret[1])
