# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def mirrorTree(self, root: TreeNode) -> TreeNode:
        def dfs(node):
            if not node:
                return
            dfs(node.left)
            dfs(node.right)
            node.left, node.right = node.right, node.left
        dfs(root)
        return root