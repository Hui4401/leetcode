class Solution:
    def partitionLabels(self, S: str) -> List[int]:
        far = [0 for _ in range(26)]
        for i, s in enumerate(S):
            far[ord(s)-97] = i
        res = []
        l = r = 0
        for i, s in enumerate(S):
            r = max(r, far[ord(s)-97])
            if i == r:
                res.append(r-l+1)
                l = r + 1
        return res