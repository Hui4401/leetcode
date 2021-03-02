from typing import List


class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        if not nums:
            return 0
        i, j , n = 0, len(nums) - 1, len(nums)
        while i != j:
            if nums[i] != val:
                i += 1
                continue
            if nums[j] == val:
                j -= 1
                n -= 1
                continue
            nums[i] = nums[j]
            j -= 1
            n -= 1
        if nums[i] == val:
            n -= 1
        return n