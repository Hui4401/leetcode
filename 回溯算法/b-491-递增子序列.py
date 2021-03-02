class Solution:
    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        res = []
        path = []
        n = len(nums)
        def dfs(start_index, len_path):
            if len_path > 1:
                res.append(path.copy())
            mset = set()
            for i in range(start_index, n):
                if path and nums[i] < path[-1] or nums[i] in mset:
                    continue
                mset.add(nums[i])
                path.append(nums[i])
                dfs(i+1, len_path+1)
                path.pop(-1)
        dfs(0, 0)
        return res