class Solution:
    def is_valid(self, row, col, n, path):
        # 检查列
        for i in range(row):
            if path[i][col] == 'Q':
                return False
        # 检查左上斜边
        i, j = row - 1, col - 1
        while i >= 0 and j >= 0:
            if path[i][j] == 'Q':
                return False
            i -= 1
            j -= 1
        # 检查右上斜边
        i, j = row - 1, col + 1
        while i >= 0 and j < n:
            if path[i][j] == 'Q':
                return False
            i -= 1
            j += 1
        return True

    def solveNQueens(self, n: int) -> List[List[str]]:
        res = []
        path = [['.' for _ in range(n)] for _ in range(n)]
        def dfs(row):
            if row == n:
                tmp = []
                for i in path:
                    tmp.append(''.join(i))
                res.append(tmp)
                return
            for col in range(n):
                if self.is_valid(row, col, n, path):
                    path[row][col] = 'Q'
                    dfs(row+1)
                    path[row][col] = '.'
        dfs(0)
        return res