class Solution:
    def minNumber(self, nums: List[int]) -> str:
        tmp = [str(s) for s in nums]
        def qsort(l, r):
            if l >= r:
                return
            i, j = l, r
            while i < j:
                while tmp[j] + tmp[l] >= tmp[l] + tmp[j] and i < j:
                    j -= 1
                while tmp[i] + tmp[l] <= tmp[l] + tmp[i] and i < j:
                    i += 1
                tmp[i], tmp[j] = tmp[j], tmp[i]
            tmp[l], tmp[i] = tmp[i], tmp[l]
            qsort(l, i-1)
            qsort(i+1, r)
        qsort(0, len(tmp)-1)
        return ''.join(tmp)