class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        if len(nums) == 1:
            return
        index = len(nums) - 2
        while index >= 0:
            if nums[index] >= nums[index+1]:
                index -= 1
            else:
                break
        if index < 0:
            nums[:] = nums[::-1]
            return

        i = len(nums) - 1
        while nums[i] <= nums[index]:
            i -= 1
        nums[index], nums[i] = nums[i], nums[index]
        nums[index+1:] = nums[index+1:][::-1]