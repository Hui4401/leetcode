# 笨比解法
# class Solution:
#     def add_len1_max(self, num1: str, num2: str) -> str:
#         num1 = num1[::-1]
#         num2 = num2[::-1]
#         r = ''
#         carry = 0
#         for i in range(0, len(num2)):
#             tmp = int(num1[i]) + int(num2[i]) + carry
#             if tmp >= 10:
#                 carry = 1
#                 tmp -= 10
#             else:
#                 carry = 0
#             r += str(tmp)
#         for i in range(len(num2), len(num1)):
#             tmp = int(num1[i]) + carry
#             if tmp >= 10:
#                 carry = 1
#                 tmp -= 10
#             else:
#                 carry = 0
#             r += str(tmp)
#         if carry == 1:
#             r += '1'
#         return r[::-1]

#     def addStrings(self, num1: str, num2: str) -> str:
#         len1, len2 = len(num1), len(num2)
#         return self.add_len1_max(num1, num2) if len1 >= len2 else self.add_len1_max(num2, num1)


class Solution:
    def addStrings(self, num1: str, num2: str) -> str:
        ret = ''
        i, j = len(num1) - 1, len(num2) - 1
        carry = 0
        while(i >= 0 or j >= 0 or carry != 0):
            # 有一个遍历完且没有进位，直接拼接剩下的
            if i < 0 and carry == 0:
                return num2[0:j+1] + ret
            if j < 0 and carry == 0:
                return num1[0:i+1] + ret

            if i >= 0:
                carry += int(num1[i])
            if j >= 0: 
                carry += int(num2[j])

            ret = str(carry % 10) + ret
            carry = carry // 10

            i -= 1
            j -= 1

        return ret


s = Solution()
print(s.addStrings('9', '999'))