class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        res = []
        path = []
        nums.sort()
        n = len(nums)
        used = [False for _ in range(n)]
        def dfs(snum):
            if snum == n:
                res.append(path.copy())
            for i in range(n):
                if used[i] or i > 0 and nums[i] == nums[i-1] and used[i-1] == False:
                    continue
                used[i] = True
                path.append(nums[i])
                dfs(snum+1)
                path.pop(-1)
                used[i] = False
        dfs(0)
        return res