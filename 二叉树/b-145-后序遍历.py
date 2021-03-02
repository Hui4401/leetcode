# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
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
            # 先压左再压右，出栈时就是右左顺序
            stack.append(node.left)
            stack.append(node.right)
        # 中右左 -> 左右中
        return res[::-1]