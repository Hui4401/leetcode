class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        res = [[0 for _ in range(n)] for _ in range(n)]
        # 行首尾，列首尾
        up, down, left, right = 0, n-1, 0, n-1
        # 0上1右2下3左
        flag = 0
        # 要填的数，每填一个+1
        num = 1
        while up <= down and left <= right:
            flag = flag % 4
            if flag == 0:
                for i in range(left, right+1):
                    res[up][i] = num
                    num += 1
                up += 1
            elif flag == 1:
                for i in range(up, down+1):
                    res[i][right] = num
                    num += 1
                right -= 1
            elif flag == 2:
                for i in range(right, left-1, -1):
                    res[down][i] = num
                    num += 1
                down -= 1
            else:
                for i in range(down, up-1, -1):
                    res[i][left] = num
                    num += 1
                left += 1
            flag += 1
        return res