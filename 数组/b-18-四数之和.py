from typing import List


class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        n = len(nums)
        if n < 4:
            return []
        nums.sort()
        res = []
        for l in range(n-3):
            # target小于0时不能判断，因为继续加负数总和会减少
            if target >= 0 and nums[l] > target:
                return res
            if l > 0 and nums[l] == nums[l-1]:
                # l去重
                continue
            for i in range(l+1, n-2):
                if i > l+1 and nums[i] == nums[i-1]:
                    # i去重
                    continue
                j = i + 1
                k = n - 1
                while j < k:
                    s = nums[l] + nums[i] + nums[j] + nums[k]
                    if s < target:
                        j += 1
                    elif s > target:
                        k -= 1
                    else:
                        res.append([nums[l], nums[i], nums[j], nums[k]])
                        j += 1
                        k -= 1
                        # j,k 去重
                        while j<k and nums[j] == nums[j-1]:
                            j += 1
                        while j<k and nums[k] == nums[k+1]:
                            k -= 1
        return res