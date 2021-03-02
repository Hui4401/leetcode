class Solution:
    def is_valid(self, board, row, col, num):
        # 行
        for i in range(9):
            if board[row][i] == num:
                return False
        # 列
        for i in range(9):
            if board[i][col] == num:
                return False
        # 3*3
        si, sj = row // 3, col //3
        for i in range(si*3, si*3+3):
            for j in range(sj*3, sj*3+3):
                if board[i][j] == num:
                    return False
        return True
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        def dfs(row, com):
            for row in range(9):
                for col in range(9):
                    if board[row][col] != '.':
                        continue
                    for i in range(1, 10):
                        num = str(i)
                        if self.is_valid(board, row, col, num):
                            board[row][col] = num
                            if dfs(row+1, com+1):
                                return True
                            board[row][col] = '.'
                    return False
            return True
        dfs(0, 0)