# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# class Solution:
#     def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
#         def find_min(root):
#             if not root:
#                 return
#             left = find_min(root.left)
#             return left if left else root.val
#         def dfs(node, key):
#             if not node:
#                 return None
#             if key < node.val:
#                 node.left = dfs(node.left, key)
#                 return node
#             if key > node.val:
#                 node.right = dfs(node.right, key)
#                 return node
#             if not node.left:
#                 return node.right
#             if not node.right:
#                 return node.left
#             minn = find_min(node.right)
#             node.val = minn
#             node.right = dfs(node.right, minn)
#             return node
#         return dfs(root, key)

class Solution:
    def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
        def dfs(node):
            if not node:
                return None
            if key < node.val:
                node.left = dfs(node.left)
                return node
            if key > node.val:
                node.right = dfs(node.right)
                return node
            if not node.left:
                return node.right
            if not node.right:
                return node.left
            p = node.right
            while p.left:
                p = p.left
            p.left = node.left
            return node.right
        return dfs(root)
