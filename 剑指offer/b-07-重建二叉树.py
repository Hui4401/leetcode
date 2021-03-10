# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        def dfs(pre, ino):
            if not pre:
                return None
            mid = pre[0]
            index = ino.index(mid)
            node = TreeNode(mid)
            node.left = dfs(pre[1:1+index], ino[:index])
            node.right = dfs(pre[1+index:], ino[index+1:])
            return node
        return dfs(preorder, inorder)