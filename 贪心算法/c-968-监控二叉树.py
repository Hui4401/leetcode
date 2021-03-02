# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def minCameraCover(self, root: TreeNode) -> int:
        res = 0
        # 0 未覆盖；1 已覆盖；2 已装摄像头
        def dfs(node):
            nonlocal res
            if not node:
                return 1

            left = dfs(node.left)
            right = dfs(node.right)

            # 左右孩子都已覆盖，则当前节点未覆盖
            if left == 1 and right == 1:
                return 0

            # 左右孩子有一个未覆盖，则当前节点应该安装摄像头
            if left == 0 or right == 0:
                res += 1
                return 2

            # 左右孩子有一个有摄像头，则当前节点已覆盖
            if left == 2 or right == 2:
                return 1

        if dfs(root) == 0:
            res += 1
        return res
