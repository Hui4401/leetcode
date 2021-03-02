# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:

        if not root:
            return []

        queue = [root]
        ret = []

        while queue:
            tmp_queue = []
            tmp_ret = []
            for i in queue:
                tmp_ret.append(i.val)
                if i.left:
                    tmp_queue.append(i.left)
                if i.right:
                    tmp_queue.append(i.right)
            queue = tmp_queue
            ret.insert(0, tmp_ret)

        return ret