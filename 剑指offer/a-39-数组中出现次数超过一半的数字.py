class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        major, num = 0, 0
        for i in range(len(nums)):
            if num == 0:
                major = nums[i]
                num = 1
            elif nums[i] != major:
                num -= 1
            else:
                num += 1
        return major