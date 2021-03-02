# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def constructMaximumBinaryTree(self, nums: List[int]) -> TreeNode:
        def dfs(nums):
            if not nums:
                return
            maxn = -2**31
            index = -1
            for i, num in enumerate(nums):
                if num > maxn:
                    maxn = num
                    index = i
            node = TreeNode(maxn)
            node.left = dfs(nums[:index])
            node.right = dfs(nums[index+1:])
            return node
        return dfs(nums)