class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        # 构造二维矩阵
        res = []
        for _ in range(n):
            res.append([None for _ in range(n)])

        loop = n // 2           # 要填几圈
        startx = starty = 0		# 每一圈的起始位置
        offset = n - 1			# 当前圈每条边负责填的块数
        count = 1				# 递增，代表填入的数据

        while loop:
            i, j = startx, starty
			# 填上边
            for j in range(starty, starty+offset):
                res[i][j] = count
                count += 1
            # 填右边
            j += 1
            for i in range(startx, startx+offset):
                res[i][j] = count
                count += 1
            # 填下边
            i += 1
            for j in range(j, starty, -1):
                res[i][j] = count
                count += 1
            # 填左边
            j -= 1
            for i in range(i, startx, -1):
                res[i][j] = count
                count += 1
            startx += 1
            starty += 1
            loop -= 1
            offset -= 2			# 这里注意下一圈每条边负责的块是要减2而不是减1的
        # 奇数要单独填中心
        if n % 2 == 1:
            mid = n // 2
            res[mid][mid] = count
        return res