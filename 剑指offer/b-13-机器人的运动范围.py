# class Solution:
#     def movingCount(self, m: int, n: int, k: int) -> int:
#         sumi = [i//10 + i%10 for i in range(m)]
#         sumj = [j//10 + j%10 for j in range(n)]
#         # 可以走的位置
#         board = [[True for _ in range(n)] for _ in range(m)]
#         for i in range(m):
#             for j in range(n):
#                 if sumi[i] + sumj[j] > k:
#                     # 不可走
#                     board[i][j] = False
#         res = 0
#         def dfs(i, j):
#             nonlocal res
#             res += 1
#             board[i][j] = False
#             # 向右
#             if j < n-1 and board[i][j+1]:
#                 dfs(i, j+1)
#             # 向下
#             if i < m-1 and board[i+1][j]:
#                 dfs(i+1, j)
#         dfs(0, 0)
#         return res

from collections import deque
class Solution:
    def movingCount(self, m: int, n: int, k: int) -> int:
        sumi = [i//10 + i%10 for i in range(m)]
        sumj = [j//10 + j%10 for j in range(n)]
        # 可以走的位置
        board = [[True for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if sumi[i] + sumj[j] > k:
                    # 不可走
                    board[i][j] = False
        res = 0
        queue = deque()
        queue.append((0, 0))
        while queue:
            i, j = queue.popleft()
            if not board[i][j]:
                continue
            res += 1
            board[i][j] = False
            if j < n-1 and board[i][j+1]:
                queue.append((i, j+1))
            if i < m-1 and board[i+1][j]:
                queue.append((i+1, j))
        return res