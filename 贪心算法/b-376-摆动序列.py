class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 1:
            return n
        res = 1
        # 0初始状态，1上升，2下降
        status = 0
        pre = nums[0]
        for i in range(1, n):
            if nums[i] == pre:
                continue
            elif nums[i] > pre:
                if status == 0 or status == 2:
                    status = 1
                    pre = nums[i]
                    res += 1
                else:
                    # 持续上升，选择最新的
                    pre = nums[i]
            else:
                if status == 0 or status == 1:
                    status = 2
                    pre = nums[i]
                    res += 1
                else:
                    # 持续下降，选择最新的
                    pre = nums[i]
        return res