# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def convertBST(self, root: TreeNode) -> TreeNode:
        a = 0
        def dfs(node):
            if not node:
                return
            nonlocal a
            dfs(node.right)
            a = node.val = node.val + a
            dfs(node.left)
        dfs(root)
        return root