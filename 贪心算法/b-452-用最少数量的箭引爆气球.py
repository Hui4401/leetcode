class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        if not points:
            return 0
        points.sort(key=lambda x: (x[0], x[1]))
        res = 1
        site = points[0][1]
        for i in range(1, len(points)):
            if points[i][0] > site:
                res += 1
                site = points[i][1]
            else:
                site = min(site, points[i][1])
        return res
