class Solution:
    def minOperations(self, boxes: str) -> List[int]:
        tmp = []
        for i in range(len(boxes)):
            if boxes[i] == '1':
                tmp.append(i)
        bool_tmp = []
        for i in tmp:
            if i == 0:
                bool_tmp.append(False)
            else:
                bool_tmp.append(True)
        res = []
        for _ in range(len(boxes)):
            sumn = 0
            for i in range(len(tmp)):
                sumn += tmp[i]
                if bool_tmp[i]:
                    tmp[i] -= 1
                else:
                    tmp[i] += 1
                if tmp[i] == 0:
                    bool_tmp[i] = False
            res.append(sumn)
        return res
