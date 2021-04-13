class Solution:
    def smallestK(self, arr: List[int], k: int) -> List[int]:
        nums = arr
        if k == 0:
            return []
        def sort(l, r):
            tmp = nums[l]
            while l < r:
                while l < r and nums[r] >= tmp:
                    r -= 1
                nums[l] = nums[r]
                while l < r and nums[l] <= tmp:
                    l += 1
                nums[r] = nums[l]
            nums[l] = tmp
            return l
        l, r = 0, len(nums)-1
        while l <= r:
            mid = sort(l, r)
            if mid == k-1:
                return nums[:mid+1]
            elif mid > k-1:
                r = mid - 1
            else:
                l = mid + 1
        return -1
