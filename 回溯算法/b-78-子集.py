class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = []
        path = []
        n = len(nums)
        def dfs(start_index):
            res.append(path.copy())
            for i in range(start_index, n):
                path.append(nums[i])
                dfs(i+1)
                path.pop(-1)
        dfs(0)
        return res