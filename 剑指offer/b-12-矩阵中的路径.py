class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        m = len(board)
        n = len(board[0])
        # 标记哪个点可以去
        flag = [[True for _ in range(n)] for _ in range(m)]
        nw = len(word)
        def dfs(i, j, index):
            # 后面递归时确保不会越界
            if board[i][j] != word[index]:
                return False
            if index == nw - 1:
                return True

            flag[i][j] = False
            if j > 0 and flag[i][j-1]:
                if dfs(i, j-1, index+1):
                    return True
            if i > 0 and flag[i-1][j]:
                if dfs(i-1, j, index+1):
                    return True
            if j < n-1 and flag[i][j+1]:
                if dfs(i, j+1, index+1):
                    return True
            if i < m-1 and flag[i+1][j]:
                if dfs(i+1, j, index+1):
                    return True
            flag[i][j] = True
            return False

        for i in range(m):
            for j in range(n):
                if dfs(i, j, 0):
                    return True
        return False
