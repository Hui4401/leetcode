class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        maps = ['abc', 'def', 'ghi', 'jkl', 'mno', 'pqrs', 'tuv', 'wxyz']
        res = []
        def dfs(s, index):
            if index == len(digits):
                res.append(s)
                return
            digit = int(digits[index])
            strings = maps[digit-2]
            for i in strings:
                dfs(s+i, index+1)
        if not digits:
            return res
        dfs('', 0)
        return res