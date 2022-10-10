### 12-矩阵中的路径

dfs

```python
def exist(self, board: List[List[str]], word: str) -> bool:
    m = len(board)
    n = len(board[0])
    # 标记哪个点可以去
    flag = [[True for _ in range(n)] for _ in range(m)]
    nw = len(word)
    def dfs(i, j, index):
        # 后面递归时保证不会越界
        if board[i][j] != word[index]:
            return False
        if index == nw - 1:
            return True

        flag[i][j] = False
        if j > 0 and flag[i][j-1]:
            if dfs(i, j-1, index+1):
                return True
        if i > 0 and flag[i-1][j]:
            if dfs(i-1, j, index+1):
                return True
        if j < n-1 and flag[i][j+1]:
            if dfs(i, j+1, index+1):
                return True
        if i < m-1 and flag[i+1][j]:
            if dfs(i+1, j, index+1):
                return True
        flag[i][j] = True
        return False

    for i in range(m):
        for j in range(n):
            if dfs(i, j, 0):
                return True
    return False
```

### 13-机器人的运动范围

dfs，从左上角固定开始的话其实只向右向下走就能走遍地图，且走过的不用再走

```python
def movingCount(self, m: int, n: int, k: int) -> int:
    sumi = [i//10 + i%10 for i in range(m)]
    sumj = [j//10 + j%10 for j in range(n)]
    # 可以走的位置
    board = [[True for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            if sumi[i] + sumj[j] > k:
                # 不可走
                board[i][j] = False
    res = 0
    def dfs(i, j):
        nonlocal res
        res += 1
        board[i][j] = False
        # 向右
        if j < n-1 and board[i][j+1]:
            dfs(i, j+1)
        # 向下
        if i < m-1 and board[i+1][j]:
            dfs(i+1, j)
    dfs(0, 0)
    return res
```

bfs

```python
def movingCount(self, m: int, n: int, k: int) -> int:
    sumi = [i//10 + i%10 for i in range(m)]
    sumj = [j//10 + j%10 for j in range(n)]
    # 可以走的位置
    board = [[True for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            if sumi[i] + sumj[j] > k:
                # 不可走
                board[i][j] = False
    res = 0
    queue = deque()
    queue.append((0, 0))
    while queue:
        i, j = queue.popleft()
        if not board[i][j]:
            continue
        res += 1
        board[i][j] = False
        if j < n-1 and board[i][j+1]:
            queue.append((i, j+1))
        if i < m-1 and board[i+1][j]:
            queue.append((i+1, j))
    return res
```

### 14-1- 剪绳子

与拆分数字一样，可以动规：dp[i] = max(dp[i], max(j\*(i-j), j\*dp[i-j]))

其实只要在大于4时尽可能拆分出3即可：

```python
def cuttingRope(self, n: int) -> int:
    if n <= 3:
        return n-1
    a = n // 3
    b = n % 3
    if b == 1:
        return (3 ** (a-1)) * 4
    elif b == 0:
        return 3 ** a
    return (3 ** a) * b
```

### 26-树的子结构

子结构不是子树

```python
def isSubStructure(self, A: TreeNode, B: TreeNode) -> bool:
    # 判断 node2 是不是 node1 的子结构
    def dfs(node1, node2):
        # 由于约定空树不是任何树的子结构，所以有任何一个为空返回false
        if not node1 or not node2:
            return False
        if node1.val != node2.val:
            return False
        # node2 到头了，则这一部分是子结构
        if not node2.left and not node2.right:
            return True
        # 走 node2 有孩子的一边
        b1 = b2 = True
        if node2.left:
            b1 = dfs(node1.left, node2.left)
        if node2.right:
            b2 = dfs(node1.right, node2.right)
        return True if b1 and b2 else False
    # 任意一种方式遍历A，不断判断B是不是A子树的子结构
    queue = deque([A])
    while queue:
        size = len(queue)
        for _ in range(size):
            node = queue.popleft()
            if dfs(node, B):
                return True
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    return False
```

### 39-出现次数超过一半的数

- 出现此处超过一半，则主元素的数量减去其他所有元素的数量还要 >= 1
- 主元素数量为0，下一个元素为主元素，数量为1，否则，若当前元素和主元素不相等，主元素数量-1，相等+1

```python
def majorityElement(self, nums: List[int]) -> int:
    major, num = 0, 0
    for i in range(len(nums)):
        if num == 0:
            major = nums[i]
            num = 1
        elif nums[i] != major:
            num -= 1
        else:
            num += 1
    return major
```

### 40-最小k个数

- 用堆，但不用构造所有数
- 构造大小为 k 的 大顶堆，然后只有进来的元素比堆顶小时弹出堆顶元素然后加入堆
- python只有小顶堆，元素取反，比堆顶大时操作

```python
def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
    if k == 0:
        return []
    heap = [-i for i in arr[:k]]
    heapq.heapify(heap)
    for i in range(k, len(arr)):
        if -arr[i] > heap[0]:
            heapq.heapreplace(heap, -arr[i])
    res = []
    while len(heap):
        res.append(-heapq.heappop(heap))
    return res
```

### 41-数据流中位数

- 一个大顶堆一个小顶堆，只要保证两者容量差不超过1（固定让一个多），就可以快速找出中间的数
- 要注意每进来一个数都要在两个堆里走一遍，否则不能保证中间有序，比如1,3；2

```python
import heapq
class MedianFinder:
    def __init__(self):
        self.heapmax = []
        self.heapmin = []
        heapq.heapify(self.heapmax)
        heapq.heapify(self.heapmin)

    def addNum(self, num: int) -> None:
        heapq.heappush(self.heapmax, -num)
        heapq.heappush(self.heapmin, -heapq.heappop(self.heapmax))
        # 这里固定大顶堆比小顶堆多
        if len(self.heapmin) - len(self.heapmax) >= 1:
            heapq.heappush(self.heapmax, -heapq.heappop(self.heapmin))

    def findMedian(self) -> float:
        if len(self.heapmax) == len(self.heapmin):
            return (self.heapmin[0] - self.heapmax[0]) / 2
        return -self.heapmax[0]
```

### 45-把数组排成最小数

- 转为字符串，按 a+b < b+a 排序

```python
class Solution:
    def minNumber(self, nums: List[int]) -> str:
        tmp = [str(s) for s in nums]
        def qsort(l, r):
            if l >= r:
                return
            i, j = l, r
            flag = tmp[i]
            while i < j:
                while i < j and tmp[j] + flag >= flag + tmp[j]:
                    j -= 1
                tmp[i] = tmp[j]
                while i < j and tmp[i] + flag <= flag + tmp[i]:
                    i += 1
                tmp[j] = tmp[i]
            tmp[i] = flag
            qsort(l, i-1)
            qsort(i+1, r)
        qsort(0, len(tmp)-1)
        return ''.join(tmp)
```

### 48-最长不重复子串

i,j 从 0 开始，每当 j 指向的字符出现过（128数组），就从 i 指向的地方向后移动直到 j 指向的字符不再出现

```python
def lengthOfLongestSubstring(self, s: str) -> int:
    flag = [False for _ in range(128)]
    i = j = tmp = res = 0
    while j < len(s) and (len(s) - i) >= res:
        while flag[ord(s[j])]:
            flag[ord(s[i])] = False
            tmp -= 1
            i += 1
        flag[ord(s[j])] = True
        tmp += 1
        j += 1
        res = max(res, tmp)
    return res
```

### 53-2 0-n-1中缺失的数字

有序就想二分

```python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        i, j = 0, len(nums) - 1
        while i < j:
            mid = (i+j) >> 1
            if nums[mid] == mid:
                i = mid + 1
            else:
                j = mid - 1
        if i != nums[i]:
            return i
        return nums[i] + 1 
```



### 59-1-滑动窗口最大值

单调队列，每次右移，如果移出去的元素等于队列头，从队列弹出这个元素，同时从尾部弹出所有小于窗口新增元素的元素，然后把新增元素加入队列

### 59-2-队列的最大值

- 最大值出队后，如何知道下一个最大值？单调队列

```python
class MaxQueue:
    def __init__(self):
        self.queue = deque()
        self.helper = deque()

    def max_value(self) -> int:
        return self.helper[0] if self.helper else -1

    def push_back(self, value: int) -> None:
        self.queue.append(value)
        while self.helper and self.helper[-1] < value:
            self.helper.pop()
        self.helper.append(value)

    def pop_front(self) -> int:
        if not self.queue:
            return -1
        res = self.queue.popleft()
        if self.helper[0] == res:
            self.helper.popleft()
        return res
```

### 60-n个骰子的点数

- dp\[i][j] 表示投出 i 个骰子后得到总点数 j 的次数
- j 有 1到6 中可能的点数，如果是1，那么总点数为 j 说明前一个是投出 j-1 的状态
- 投 n 个骰子点数范围 n 到 6n，dp数组大小按最大来
- n 个骰子总点数 6**n，除dp\[n][i]得到概率

```python
def dicesProbability(self, n: int) -> List[float]:
    dp = [[0 for _ in range(67)] for _ in range(12)]
    for i in range(1, 7):
        dp[1][i] = 1
    for i in range(2, n+1):
        for j in range(i, 6*i+1):
            for k in range(1, 7):
                if j - k <= 0:
                    break
                dp[i][j] += dp[i-1][j-k]
    total = 6 ** n
    res = []
    for i in range(n, n*6+1):
        res.append(dp[n][i]/total)
    return res
```

### 65-不用加减乘除做加法

- a ^ b：无进位加法
- (a & b) << 1：进位
- 所以不断异或（加）进位直到进位为0
- python的整数范围是无穷大（内存的范围内），必须将运算过程限定在 32 位内（和 0xffffffff 与）（算进位时需要，因为要左移）
- 算完之后还原成原来 python 表示的数时有两种情况
- 如果第32位为0，正数（高于32位都是0，限定过了），一切正常
- 如果第32位为1，负数，那么需要将高于32位都变成1（负数形式）
- ~(a ^ 0xffffffff)，先将低32位取反，然后整个取反，结果是低32位不变，高于32都变成1

```python
def add(self, a: int, b: int) -> int:
    x = 0xffffffff
    a, b = a & x, b & x
    while b != 0:
        c = (a & b) << 1 & x
        a = a ^ b
        b = c
    return ~(a ^ x) if a & 0x80000000 else a
```

### 66-构建乘积数组

![Picture1.png](./imgs/20210320105303-1663498162583-1.png)

```python
def constructArr(self, a: List[int]) -> List[int]:
    b = [1]*len(a)
    # 下三角
    for i in range(1, len(a)):
        b[i] = b[i-1] * a[i-1]
    # 上三角
    tmp = 1
    for i in range(len(a)-2, -1, -1):
        tmp *= a[i+1]	# 记录上三角
        b[i] *= tmp		# 原下三角乘上三角
    return b
```



