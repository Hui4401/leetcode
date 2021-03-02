# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
        def dfs(ino, pos):
            if not pos:
                return None
            mid = pos[-1]
            index = ino.index(mid)
            node = TreeNode(mid)
            # 后序数组切割标准：最后一个不用，然后中序数组大小一定和后序数组长度相同
            node.left = dfs(ino[:index], pos[:index])
            node.right = dfs(ino[index+1:], pos[index:-1])
            return node
        return dfs(inorder, postorder)