class Solution:
    def subarraysDivByK(self, A: List[int], K: int) -> int:
        mp = [0 for _ in range(K)]
        mp[0] = 1
        mod = 0
        res = 0
        for a in A:
            mod = (mod + a) % K
            if mod < 0:
                mod = -mod
            res += mp[mod]
            mp[mod] += 1
        return res