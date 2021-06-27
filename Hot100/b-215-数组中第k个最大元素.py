# class Solution:
#     def findKthLargest(self, nums: List[int], k: int) -> int:
#         def build_heap(nums):
#             '''建立小顶堆'''
#             for i in range(len(nums)//2-1, -1, -1):
#                 adjust_down(nums, i)

#         def adjust_down(nums, index):
#             '''向下调整'''
#             l, r, min_index = index*2+1, index*2+2, index
#             if l < len(nums) and nums[l] < nums[min_index]:
#                 min_index = l
#             if r < len(nums) and nums[r] < nums[min_index]:
#                 min_index = r
#             if min_index != index:
#                 nums[index], nums[min_index] = nums[min_index], nums[index]
#                 adjust_down(nums, min_index)

#         def pop(nums):
#             '''弹出堆顶元素'''
#             nums[0], nums[-1] = nums[-1], nums[0]
#             res = nums.pop(-1)
#             adjust_down(nums, 0)
#             return res

#         def adjust_up(nums, index):
#             '''向上调整'''
#             parent = (index+1)//2-1
#             if parent >= 0 and nums[index] < nums[parent]:
#                 nums[index], nums[parent] = nums[parent], nums[index]
#                 adjust_up(nums, parent)

#         def push(nums, num):
#             '''添加元素'''
#             nums.append(num)
#             adjust_up(nums, len(nums)-1)

#         heap = nums[:k]
#         build_heap(heap)
#         for i in range(k, len(nums)):
#             if nums[i] > heap[0]:
#                 heap.append(nums[i])
#                 pop(heap)
#         return heap[0]

class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        def sort(l, r):
            tmp = nums[l]
            while l < r:
                while l < r and nums[r] <= tmp:
                    r -= 1
                nums[l] = nums[r]
                while l < r and nums[l] >= tmp:
                    l += 1
                nums[r] = nums[l]
            nums[l] = tmp
            return l
        l, r = 0, len(nums)-1
        while l <= r:
            mid = sort(l, r)
            if mid == k-1:
                return nums[mid]
            elif mid > k-1:
                r = mid - 1
            else:
                l = mid + 1
        return -1
