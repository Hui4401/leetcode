# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
        if not root:
            return []
        res = []
        path = []
        def dfs(node, sumn):
            path.append(node.val)
            if not node.left and not node.right and sumn+node.val == sum:
                res.append(path.copy())
                return
            if node.left:
                dfs(node.left, sumn+node.val)
                path.pop(-1)
            if node.right:
                dfs(node.right, sumn+node.val)
                path.pop(-1)
        dfs(root, 0)
        return res