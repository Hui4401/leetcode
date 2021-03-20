class Solution:
    def singleNumbers(self, nums: List[int]) -> List[int]:
        tmp = 0
        for num in nums:
            tmp = tmp ^ num
        div = 1
        while div & tmp == 0:
            div = div << 1
        a, b = 0, 0
        for num in nums:
            if num & div:
                a = a ^ num
            else:
                b = b ^ num
        return [a, b]