# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        stack = [root]
        res = []
        while stack:
            node = stack.pop(-1)
            # 节点没有左孩子或右孩子时会压入None，出栈时要判断
            if not node:
                continue
            res.append(node.val)
            # 先压右再压左，出栈时就是左右顺序
            stack.append(node.right)
            stack.append(node.left)
        return res

# # 统一写法
# class Solution:
#     def preorderTraversal(self, root: TreeNode) -> List[int]:
#         if not root:
#             return []
#         stack = [root]
#         res = []
#         while stack:
#             node = stack.pop(-1)
#             if node:
#                 # 右（空节点不入栈）
#                 if node.right:
#                     stack.append(node.right)
#                 # 左
#                 if node.left:
#                     stack.append(node.left)
#                 # 中，遍历过但没有访问，加入None标记
#                 stack.append(node)
#                 stack.append(None)
#             # 只有遇到空节点的时候，才访问下一个节点
#             else:
#                 res.append(stack.pop(-1).val)
#         return res
