class Solution:
    def verifyPostorder(self, postorder: List[int]) -> bool:
        stack = []
        root = float('inf')
        for i in range(len(postorder)-1, -1, -1):
            if postorder[i] > root:
                return False
            while stack and postorder[i] < stack[-1]:
                root = stack.pop(-1)
            stack.append(postorder[i])
        return True

# class Solution:
#     def verifyPostorder(self, postorder: [int]) -> bool:
#         def dfs(i, j):
#             if i >= j:
#                 return True
#             p = i
#             while postorder[p] < postorder[j]:
#                 p += 1
#             m = p
#             while postorder[p] > postorder[j]:
#                 p += 1
#             return p == j and dfs(i, m - 1) and dfs(m, j - 1)

#         return dfs(0, len(postorder)-1)