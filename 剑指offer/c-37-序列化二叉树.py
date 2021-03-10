# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

from collections import deque
class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        if not root:
            return ''
        res = []
        queue = deque([root])
        while queue:
            node = queue.popleft()
            if not node:
                res.append(None)
            else:
                res.append(node.val)
                queue.append(node.left)
                queue.append(node.right)
        # 去掉尾部None，不要用 not 判断，会把0也去掉
        while res[-1] == None:
            res.pop(-1)
        return str(res)


    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        # '[]'
        if len(data) < 3:
            return None
        data = data[1:-1].split(', ')
        l = []
        for i in data:
            if i == 'None':
                l.append(None)
            else:
                l.append(int(i))
        n = len(l)
        # 根节点
        root = TreeNode(l[0])
        i = 1
        # 层序
        queue = deque([root])
        while queue:
            node = queue.popleft()
            if i < n and l[i] != None:
                node.left = TreeNode(l[i])
                queue.append(node.left)
            i += 1
            if i < n and l[i] != None:
                node.right = TreeNode(l[i])
                queue.append(node.right)
            i += 1
        return root


# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.deserialize(codec.serialize(root))