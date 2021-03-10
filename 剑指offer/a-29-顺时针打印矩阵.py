class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        m = len(matrix)
        if m == 0:
            return []
        n = len(matrix[0])
        if n == 0:
            return []
        res = []
        # 行首尾，列首尾
        up, down, left, right = 0, m-1, 0, n-1
        # 0上1右2下3左
        flag = 0
        while up <= down and left <= right:
            flag = flag % 4
            if flag == 0:
                for i in range(left, right+1):
                    res.append(matrix[up][i])
                up += 1
            elif flag == 1:
                for i in range(up, down+1):
                    res.append(matrix[i][right])
                right -= 1
            elif flag == 2:
                for i in range(right, left-1, -1):
                    res.append(matrix[down][i])
                down -= 1
            else:
                for i in range(down, up-1, -1):
                    res.append(matrix[i][left])
                left += 1
            flag += 1
        return res