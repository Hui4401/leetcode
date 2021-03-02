# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        def dfs(root):
            if not root:
                return True, 0
            bl, left = dfs(root.left)
            br, right = dfs(root.right)
            if not bl or not br or left > right+1 or right > left+1:
                return False, 0
            return True, 1+max(left, right)
        b, _ = dfs(root)
        return b