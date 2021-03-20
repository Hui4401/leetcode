class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        def mergesort(l, r):
            if l >= r:
                return 0
            mid = (l + r) >> 1
            left = mergesort(l, mid)
            right = mergesort(mid+1, r)
            tmp1 = [i for i in nums[l:mid+1]]
            tmp2 = [i for i in nums[mid+1:r+1]]
            i = j = tmp = 0
            k = l
            while i < mid - l + 1 and j < r - mid:
                if tmp1[i] <= tmp2[j]:
                    nums[k] = tmp1[i]
                    i += 1
                else:
                    tmp += mid - l - i + 1
                    nums[k] = tmp2[j]
                    j += 1
                k += 1
            while i < mid - l + 1:
                nums[k] = tmp1[i]
                i += 1
                k += 1
            while j < r - mid:
                nums[k] = tmp2[j]
                j += 1
                k += 1
            return tmp + left + right
        return mergesort(0, len(nums)-1)


