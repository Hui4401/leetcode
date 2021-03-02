class Solution:
    def lemonadeChange(self, bills: List[int]) -> bool:
        rest = [0, 0]
        for bill in bills:
            if bill == 5:
                rest[0] += 1
            elif bill == 10:
                if rest[0] > 0:
                    rest[0] -= 1
                    rest[1] += 1
                else:
                    return False
            else:
                if rest[1] > 0 and rest[0] > 0:
                    rest[1] -= 1
                    rest[0] -= 1
                elif rest[0] > 2:
                    rest[0] -= 3
                else:
                    return False
        return True