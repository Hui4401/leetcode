from collections import deque
class Solution:
    def numIslands(self, grid: [[str]]) -> int:
        def bfs(i, j):
            queue = deque()
            queue.append((i, j))
            while queue:
                i, j = queue.popleft()
                if 0 <= i < len(grid) and 0 <= j < len(grid[0]) and grid[i][j] == '1':
                    grid[i][j] = '0'
                    queue.append((i+1, j))
                    queue.append((i-1, j))
                    queue.append((i, j-1))
                    queue.append((i, j+1))
        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '1':
                    bfs(i, j)
                    count += 1
        return count