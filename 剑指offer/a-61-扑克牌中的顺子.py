class Solution:
    def isStraight(self, nums: List[int]) -> bool:
        tmp = [0 for i in range(14)]
        minn, maxn = 14, -1
        for num in nums:
            if num == 0:
                continue
            if tmp[num] == 1:
                return False
            tmp[num] += 1
            if num < minn:
                minn = num
            if num > maxn:
                maxn = num
        return (maxn-minn) < 5