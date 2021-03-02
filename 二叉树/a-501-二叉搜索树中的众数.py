# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def findMode(self, root: TreeNode) -> List[int]:
        pre = None
        n = 0
        maxn = 0
        res = []

        def dfs(node):
            if not node:
                return
            nonlocal pre, n, maxn

            dfs(node.left)

            if node.val == pre:
                n += 1
            else:
                n = 1
            if n == maxn:
                res.append(node.val)
            elif n > maxn:
                res.clear()
                res.append(node.val)
                maxn = n
            pre = node.val

            dfs(node.right)

        dfs(root)
        return res
