class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        up, down = 0, len(matrix)-1
        while up < down:
            for i in range(len(matrix)):
                matrix[up][i], matrix[down][i] = matrix[down][i], matrix[up][i]
            up += 1
            down -= 1
        for i in range(len(matrix)):
            for j in range(i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
