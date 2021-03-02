class Solution:
    def is_valid(self, s):
        if not s:
            return False
        if s[0] == '0' and len(s) > 1:
            return False
        if int(s) > 255:
            return False
        return True
    def restoreIpAddresses(self, s: str) -> List[str]:
        res = []
        n = len(s)
        if n < 4 or n > 12:
            return res
        def dfs(start_index, pnum, ip):
            if pnum == 3:
                tmp = s[start_index:]
                if self.is_valid(tmp):
                    res.append(ip+tmp)
            for i in range(start_index, n):
                tmp = s[start_index:i+1]
                if not self.is_valid(tmp):
                    break
                dfs(i+1, pnum+1, ip+tmp+'.')
        dfs(0, 0, '')
        return res