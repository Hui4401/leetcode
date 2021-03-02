# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def hasPathSum(self, root: TreeNode, targetSum: int) -> bool:
        def dfs(node, sumn):
            if not node:
                return False
            if not node.left and not node.right and sumn + node.val == targetSum:
                return True
            left = dfs(node.left, sumn+node.val)
            right = dfs(node.right, sumn+node.val)
            return True if left or right else False
        return dfs(root, 0)