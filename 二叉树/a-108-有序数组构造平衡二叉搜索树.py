# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        def dfs(nums):
            if not nums:
                return None
            n = len(nums)
            if n == 1:
                return TreeNode(nums[0])
            i = n // 2
            node = TreeNode(nums[i])
            node.left = dfs(nums[:i])
            node.right = dfs(nums[i+1:])
            return node
        return dfs(nums)
