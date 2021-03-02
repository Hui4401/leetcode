class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        res = []
        path = []
        nums.sort()
        n = len(nums)
        used = [False for _ in range(n)]
        def dfs(start_index):
            res.append(path.copy())
            for i in range(start_index, n):
                if i > 0 and nums[i] == nums[i-1] and used[i-1] == False:
                    continue
                path.append(nums[i])
                used[i] = True
                dfs(i+1)
                used[i] = False
                path.pop(-1)
        dfs(0)
        return res