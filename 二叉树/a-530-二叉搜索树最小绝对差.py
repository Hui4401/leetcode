# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def getMinimumDifference(self, root: TreeNode) -> int:
        ans = ret = -1
        def dfs(node):
            if not node:
                return
            nonlocal ans, ret
            dfs(node.left)
            if ans != -1:
                sub = abs(node.val - ans)
                if sub < ret or ret == -1:
                    ret = sub
            ans = node.val
            dfs(node.right)
        dfs(root)
        return ret