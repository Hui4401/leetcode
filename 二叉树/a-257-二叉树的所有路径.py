# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# class Solution:
#     def binaryTreePaths(self, root: TreeNode) -> List[str]:
#         def dfs(root):
#             if not root:
#                 return []
#             l1 = dfs(root.left)
#             l2 = dfs(root.right)
#             res = []
#             if not l1 and not l2:
#                 res.append(str(root.val))
#             if l1:
#                 for i in l1:
#                     res.append(str(root.val)+'->'+i)
#             if l2:
#                 for i in l2:
#                     res.append(str(root.val)+'->'+i)
#             return res
#         return dfs(root)

class Solution:
    def binaryTreePaths(self, root: TreeNode) -> List[str]:
        def dfs(cur, path, res):
            path.append(str(cur.val))
            if not cur.left and not cur.right:
                res.append('->'.join(path))
                return
            if cur.left:
                dfs(cur.left, path, res)
                # 回溯要和递归永远在一起
                path.pop(-1)
            if cur.right:
                dfs(cur.right, path, res)
                path.pop(-1)
        # path记录当前节点以前的路径
        path = []
        # res是最终结果，遇到叶子节点是把path的内容拼接好加入res
        res = []
        if not root:
            return res
        dfs(root, path, res)
        return res
