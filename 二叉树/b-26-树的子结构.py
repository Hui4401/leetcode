# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isSubStructure(self, A: TreeNode, B: TreeNode) -> bool:
        # 判断 node2 是不是 node1 的子结构
        def dfs(node1, node2):
            # 由于约定空树不是任何树的子结构，所以有任何一个为空返回false
            if not node1 or not node2:
                return False
            if node1.val != node2.val:
                return False
            # node2 到头了，则这一部分是子结构
            if not node2.left and not node2.right:
                return True
            # 走 node2 有孩子的一边
            b1 = b2 = True
            if node2.left:
                b1 = dfs(node1.left, node2.left)
            if node2.right:
                b2 = dfs(node1.right, node2.right)
            return True if b1 and b2 else False
        # 任意一种方式遍历A，不断判断B是不是A子树的子结构
        queue = deque([A])
        while queue:
            size = len(queue)
            for _ in range(size):
                node = queue.popleft()
                if dfs(node, B):
                    return True
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return False