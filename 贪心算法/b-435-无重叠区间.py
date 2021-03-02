class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        if not intervals:
            return 0
        intervals.sort(key=lambda x: x[0])
        # 前一个保留的区间
        pre = intervals[0]
        res = 0
        for i in range(1, len(intervals)):
            # 发生重叠
            if intervals[i][0] < pre[1]:
                # pre保留结束位置较近的
                if intervals[i][1] < pre[1]:
                    pre = intervals[i]
                res += 1
            else:
                pre = intervals[i]
        return res