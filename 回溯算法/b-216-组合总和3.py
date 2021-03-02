class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        res = []
        path = []
        def dfs(sumn, start_index):
            if sumn > n:
                return
            if len(path) == k:
                if sumn == n:
                    res.append(path.copy())
                return
            for i in range(start_index, 9-(k-len(path))+2):
                path.append(i)
                # sumn的回溯隐藏在传参里
                dfs(sumn+i, i+1)
                path.pop(-1)
        dfs(0, 1)
        return res