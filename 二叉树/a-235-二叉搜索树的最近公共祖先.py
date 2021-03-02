# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if p.val > q.val:
            maxn = p.val
            minn = q.val
        else:
            maxn = q.val
            minn = p.val
        def dfs(node):
            if not node:
                return
            if node.val > maxn:
                return dfs(node.left)
            if node.val < minn:
                return dfs(node.right)
            return node
        return dfs(root)