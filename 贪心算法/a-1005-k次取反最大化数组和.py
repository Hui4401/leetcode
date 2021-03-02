class Solution:
    def largestSumAfterKNegations(self, A: List[int], K: int) -> int:
        A.sort()
        sumn = sum(A)
        i = 0
        while K:
            # 遇到0，无限翻转0即可，直接返回
            if A[i] == 0:
                return sumn
            # 遇到负数，直接翻转
            elif A[i] < 0:
                A[i] = -A[i]
                sumn += A[i] * 2
                i += 1
                K -= 1
            # 遇到正数，就没必要往下走了，无限翻转这个正数和它前一个翻转后的正数中较小的一个
            else:
                # 当前正数比前一个翻转后的大，回退，翻转前一个
                if A[i] > 0 and A[i] > A[i-1]:
                    i = i - 1
                else:
                    A[i] = -A[i]
                    sumn += A[i] * 2
                    K -= 1
        return sumn
