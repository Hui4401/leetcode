class Solution:
    def partition(self, s: str) -> List[List[str]]:
        res = []
        path = []
        n = len(s)
        def dfs(start_index):
            if start_index >= n:
                res.append(path.copy())
                return
            for i in range(start_index, n):
                tmp = s[start_index:i+1]
                if tmp != tmp[::-1]:
                    continue
                path.append(tmp)
                dfs(i+1)
                path.pop(-1)
        dfs(0)
        return res