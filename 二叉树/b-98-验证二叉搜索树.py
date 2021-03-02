# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        self.pre = -2**32
        def dfs(node):
            if not node:
                return True
            if not dfs(node.left):
                return False
            if node.val <= self.pre:
                return False
            else:
                self.pre = node.val
            return dfs(node.right)
        return dfs(root)