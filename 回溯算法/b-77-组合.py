class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        res = []
        path = []
        def dfs(start_index):
            if len(path) == k:
                # 记得append path的copy，不然path会变
                res.append(path.copy())
                return
            for i in range(start_index, n-(k-len(path))+2):
                path.append(i)
                dfs(i+1)
                path.pop(-1)
        dfs(1)
        return res