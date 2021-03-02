# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        cur = root
        stack = []
        res = []
        # 根节点有右孩子，出栈回来时虽然栈空但cur指向右孩子还没有入栈，所以cur不为空时也要继续
        while cur or stack:
            # 只要当前节点不为空就入栈并遍历它的左孩子
            if cur:
                stack.append(cur)
                cur = cur.left
            # 当前空了，出栈它的父节点，加入结果并指向父节点的右孩子
            else:
                cur = stack.pop(-1)
                res.append(cur.val)
                cur = cur.right
        return res