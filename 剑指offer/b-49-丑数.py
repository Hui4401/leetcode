# class Solution:
#     def nthUglyNumber(self, n: int) -> int:
#         res = 1
#         i = 1
#         q2, q3, q5 = deque([2]), deque([3]), deque([5])
#         while i < n:
#             res = min(q2[0], q3[0], q5[0])
#             i += 1
#             q2.append(res*2)
#             q3.append(res*3)
#             q5.append(res*5)
#             if q2[0] == res:
#                 q2.popleft()
#             if q3[0] == res:
#                 q3.popleft()
#             if q5[0] == res:
#                 q5.popleft()
#         return res

class Solution:
    def nthUglyNumber(self, n: int) -> int:
        dp = [0] * n
        dp[0] = 1
        i = j = k = 0
        for x in range(1, n):
            u2, u3, u5 = dp[i]*2, dp[j]*3, dp[k]*5
            dp[x] = min(u2, u3, u5)
            if dp[x] == u2:
                i += 1
            if dp[x] == u3:
                j += 1
            if dp[x] == u5:
                k += 1
        return dp[-1]