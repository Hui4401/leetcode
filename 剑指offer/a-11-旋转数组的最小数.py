class Solution:
    def minArray(self, numbers: List[int]) -> int:
        if len(numbers) < 2:
            return numbers[0] if numbers else 0
        for i in range(1, len(numbers)):
            if numbers[i] < numbers[i-1]:
                return numbers[i]
        return numbers[0]