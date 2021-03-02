class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        path = []
        n = len(candidates)
        def dfs(sumn, start_index):
            if sumn > target:
                return
            if sumn == target:
                res.append(path.copy())
                return
            for i in range(start_index, n):
                path.append(candidates[i])
                dfs(sumn+candidates[i], i)
                path.pop(-1)
        dfs(0, 0)
        return res