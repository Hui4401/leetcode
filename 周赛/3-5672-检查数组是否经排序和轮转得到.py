class Solution:
    def check(self, nums: List[int]) -> bool:
        if not nums:
            return True
        source = sorted(nums)
        n = len(nums)
        for x in range(n):
            for i in range(n):
                if source[i] != nums[(i+x) % n]:
                    break
            else:
                return True
        return False