# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        def dfs(ino, pre):
            if not pre:
                return None
            mid = pre[0]
            index = ino.index(mid)
            node = TreeNode(mid)
            # 前序数组切割标准：第一个不用，然后前序数组大小一定和中序数组长度相同
            node.left = dfs(ino[:index], pre[1:index+1])
            node.right = dfs(ino[index+1:], pre[index+1:])
            return node
        return dfs(inorder, preorder)