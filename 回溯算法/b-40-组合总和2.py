class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        res = []
        path = []
        n = len(candidates)
        used = [False for _ in range(n)]
        def dfs(sumn, start_index):
            if sumn > target:
                return
            if sumn == target:
                res.append(path.copy())
                return
            for i in range(start_index, n):
                if i > 0 and candidates[i] == candidates[i-1] and used[i-1] == False:
                    continue
                path.append(candidates[i])
                used[i] = True
                dfs(sumn+candidates[i], i+1)
                used[i] = False
                path.pop(-1)
        dfs(0, 0)
        return res