# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
        res = []
        def dfs(node, path, sumn):
            path.append(node.val)
            if not node.left and not node.right and sumn + node.val == sum:
                res.append(path.copy())
                return
            if node.left:
                dfs(node.left, path, sumn+node.val)
                path.pop(-1)
            if node.right:
                dfs(node.right, path, sumn+node.val)
                path.pop(-1)
        if not root:
            return res
        path = []
        dfs(root, path, 0)
        return res