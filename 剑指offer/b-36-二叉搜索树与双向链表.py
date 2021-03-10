"""
# Definition for a Node.
class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
"""
class Solution:
    def treeToDoublyList(self, root: 'Node') -> 'Node':
        if not root:
            return None
        # 初始状态，要求不能创建任何新节点，不然可以搞个虚拟头结点更好写
        head = pre = None
        # 非递归遍历统一写法之中序遍历
        stack = [root]
        while stack:
            node = stack.pop(-1)
            if node:
                if node.right:
                    stack.append(node.right)
                stack.append(node)
                stack.append(None)
                if node.left:
                    stack.append(node.left)
            else:
                node = stack.pop(-1)
                if not head:
                    head = pre = node
                else:
                    pre.right = node
                    node.left = pre
                    pre = node
        # 处理首尾
        pre.right = head
        head.left = pre
        return head