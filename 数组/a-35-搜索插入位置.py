# # 递归
# class Solution:
#     def searchInsert(self, nums, target: int) -> int:
#         def search(nums, left, right):
#             # 左右闭区间时二分mid永远取下界，left>right的唯一可能就是left=mid时right=mid-1<left
#             if left > right:
#                 return left
#             mid = (right + left) // 2
#             if  target == nums[mid]:
#                 return mid
#             elif target < nums[mid]:
#                 return left if left == right else search(nums, left, mid-1)
#             else:
#                 return left+1 if left == right else search(nums, mid+1, right)
#         return search(nums, 0, len(nums)-1)


class Solution:
    def searchInsert(self, nums, target: int) -> int:
        left = 0
        right = len(nums) - 1
        while(left <= right):
            mid = (left + right) // 2
            if target < nums[mid]:
                right = mid - 1
            elif target > nums[mid]:
                left = mid + 1
            else:
                return mid
        # 左右闭区间时二分mid永远取下界，left>right的唯一可能就是left=mid时right=mid-1<left
        return left


s = Solution()
print(s.searchInsert([3, 5, 7, 9, 10], 8))