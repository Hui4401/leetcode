class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        path = []
        n = len(nums)
        used = [False for _ in range(n)]
        def dfs(snum):
            if snum == n:
                res.append(path.copy())
            for i in range(n):
                if used[i]:
                    continue
                used[i] = True
                path.append(nums[i])
                dfs(snum+1)
                path.pop(-1)
                used[i] = False
        dfs(0)
        return res