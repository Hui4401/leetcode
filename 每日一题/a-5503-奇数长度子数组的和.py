class Solution:
    def sumOddLengthSubarrays(self, arr: List[int]) -> int:
        if not arr:
            return 0

        # [1, 4, 2, 5, 3]
        ret = 0
        ji = 1
        lena = len(arr)
        while ji <= lena:
            flag = 0
            while flag + ji <= lena:
                li = arr[flag: flag+ji]
                ret += sum(li)
                flag += 1
            ji += 2
        return ret