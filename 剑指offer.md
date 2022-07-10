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
        # 后面递归时确保不会越界
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

与拆分数字一样，可以动规：dp[i] = max(dp[i], max(j\*(i-j), dp[i-j]*j))

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

### 19-正则表达式匹配

dp\[i][j] 表示 s 前 i 位和 p 前 j 位能否匹配

- i 或 j 表示空字符的状态，所以dp数组长度要比字符串长度大1
- dp\[i][j] 对应的字符是 s[i-1] 和 p[j-1]

那么计算dp\[i][j]时首先看 p[j-1]：

- p[j-1] 不为 * ，只有以下两种情况可以匹配：
  - dp\[i-1][j-1] 且 p[j-1]\==s[i-1]，即 p 第 j 位和 s 第 i 位相等
  - dp\[i-1][j-1] 且 p[j-1]=='.' ，即 p 第 j 位为 . 可以随便与 s 第 i 位匹配
- p[j-1] 为 * ，只有以下三种情况可以匹配
  - dp\[i][j-2]，即认为 p[j-1] 和 p[j-2] 出现 0 次，看前面的是否匹配
  - dp\[i-1][j] 且 p[j-2]\==s[i-1]，即利用 * 让 p[j-2] 位多出现一次看是否与 s[i-1] 位匹配
  - dp\[i-1][j] 且 p[j-2]\=='.'，即利用 * 让前一位 . 多出现一次，随便匹配

初始化：

- s，p 都为空时是匹配的

- s 为空时 p 的偶数位必须都为 * 才能匹配
- p 为空时不能匹配

```python
def isMatch(self, s: str, p: str) -> bool:
    m, n = len(s), len(p)
    # 0表示空串，数组要长一位
    dp = [[False for _ in range(n+1)] for _ in range(m+1)]

    # 初始化
    dp[0][0] = True
    for i in range(2, n+1, 2):
        if dp[0][i-2] and p[i-1] == '*':
            dp[0][i] = True                                                                                         
    for i in range(1, m+1):
        for j in range(1, n+1):
            # p[j-1] 不为 * 只有两种情况下可以匹配
            if p[j-1] != '*':
                if dp[i-1][j-1] and p[j-1] == s[i-1]:
                    dp[i][j] = True
                elif dp[i-1][j-1] and p[j-1] == '.':
                    dp[i][j] = True
            # p[j-1] 为 *，则有三种情况
            else:
                if dp[i][j-2]:
                    dp[i][j] = True
                elif dp[i-1][j] and p[j-2] == s[i-1]:
                    dp[i][j] = True
                elif dp[i-1][j] and p[j-2] == '.':
                    dp[i][j] = True
    return dp[-1][-1]
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

### 29-顺时针打印矩阵

- up，down，left，right 初始指向行列首尾
- 填完上边后up下移（+1），填完右边后right左移（-1）。。。直到 up>down 或 left>right，填充完毕
- 填数或者读都是一样道理

```python
def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
    if len(matrix) == 0:
        return []
    if len(matrix[0]) == 0:
        return []
    res = []
    # 行首尾，列首尾
    up, down, left, right = 0, len(matrix)-1, 0, len(matrix[0])-1
    # 0上1右2下3左
    flag = 0
    while up <= down and left <= right:
        flag = flag % 4
        if flag == 0:
            for i in range(left, right+1):
                res.append(matrix[up][i])
            up += 1
        elif flag == 1:
            for i in range(up, down+1):
                res.append(matrix[i][right])
            right -= 1
        elif flag == 2:
            for i in range(right, left-1, -1):
                res.append(matrix[down][i])
            down -= 1
        else:
            for i in range(down, up-1, -1):
                res.append(matrix[i][left])
            left += 1
        flag += 1
    return res
```

### 33-二叉搜索树的后序序列

- 单调栈法（on）

![image-20210307232753468](https://cdn.jsdelivr.net/gh/Hui4401/imgbed/img/2021/03/07/20210307232755.png)

```python
def verifyPostorder(self, postorder: List[int]) -> bool:
    # 单调栈，存储 递增 序列
    stack = []
    # root 初始值为正无穷大，可把树的根节点看为此无穷大节点的左孩子
    root = float('inf')
    # 倒序遍历
    for i in range(len(postorder)-1, -1, -1):
        # ri大于root，不满足
        if postorder[i] > root:
            return False
        # 寻找大于且最接近 ri 的节点（在栈底，如果栈不空的话）
        while stack and stack[-1] > postorder[i]
            root = stack.pop(-1)
        # 当前节点入栈
        stack.append(postorder[i])
    return True
```

- 递归分治（on2）

![image-20210308142941106](https://cdn.jsdelivr.net/gh/Hui4401/imgbed/img/2021/03/08/20210308142942.png)

```python
def verifyPostorder(self, postorder: [int]) -> bool:
    def dfs(i, j):
        if i >= j: 
            return True
        p = i
        while postorder[p] < postorder[j]: 
            p += 1
        m = p
        while postorder[p] > postorder[j]: 
            p += 1
        return p == j and dfs(i, m - 1) and dfs(m, j - 1)

    return dfs(0, len(postorder)-1)
```

### 34-二叉树中和为某一值的路径

```python
def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
    # 先处理空树
    if not root:
        return []
    res = []
    path = []
    def dfs(node, sumn):
        # 每到一层先把节点值加进路径
        path.append(node.val)
        # 是否到叶子节点且满足条件
        if not node.left and not node.right and sumn+node.val == sum:
            res.append(path.copy())
            return
        if node.left:
            dfs(node.left, sumn+node.val)
            path.pop(-1)
        if node.right:
            dfs(node.right, sumn+node.val)
            path.pop(-1)
    dfs(root, 0)
    return res
```

### 35-复杂链表的复制

- 可以用hash（dict）存储原节点和复制节点的映射，这样第二轮就可用 d[p].next = d.get(p.next) 和 d[p].random = d.get(p.random) 来构建连接
- 构建拼接链表，空间复杂度更低（o1）

```python
def copyRandomList(self, head: 'Node') -> 'Node':
    if not head:
        return None
    # 构建拼接节点
    p = head
    while p:
        node = Node(p.val, next = p.next)
        p.next = node
        p = node.next
    # 构建各新节点的 random 指向
    p = head
    while p:
        if p.random:
            p.next.random = p.random.next
        p = p.next.next
    # 拆分节点
    pre = head
    cur = head.next
    p = cur
    # 注意末尾
    while cur.next:
        pre.next = pre.next.next
        cur.next = cur.next.next
        pre, cur = pre.next, cur.next
    pre.next = None
    return p
```

### 37-序列化二叉树

- 非完全二叉树，不能用下标递归构造
- 按层构建，节点出队，构造节点的左孩子并入队，构造右孩子并入队

```python
from collections import deque
class Codec:
    # 序列化
    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        if not root:
            return ''
        res = []
        queue = deque([root])
        while queue:
            node = queue.popleft()
            if not node:
                res.append(None)
            else:
                res.append(node.val)
                queue.append(node.left)
                queue.append(node.right)
        # 去掉尾部None，不要用 not 判断，会把0也去掉
        while res[-1] == None:
            res.pop(-1)
        return str(res)
	# 反序列化
    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        # '[]'
        if len(data) < 3:
            return None
        data = data[1:-1].split(', ')
        l = []
        for i in data:
            if i == 'None':
                l.append(None)
            else:
                l.append(int(i))
        n = len(l)
        # 根节点
        root = TreeNode(l[0])
        i = 1
        # 层序
        queue = deque([root])
        while queue:
            node = queue.popleft()
            if i < n and l[i] != None:
                node.left = TreeNode(l[i])
                queue.append(node.left)
            i += 1
            if i < n and l[i] != None:
                node.right = TreeNode(l[i])
                queue.append(node.right)
            i += 1
        return root
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
            heapq.heappop(heap)
            heapq.heappush(heap, -arr[i])
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

### 43-1-1~n整数中1的个数

- 将 n 写为 nxnx−1⋯ n2n1，将各位出现1的次数加起来，注意从后往前每次只关心当前位
- 将 n 按位分为三段，high，cur，low，cur为当前位
- 那么初始是个位，high为n//10，cur为n%10，low设为0，digit代表数位为1
- 那么每次向前，low+=cur*digit，cur=high%10，high//=10，digit\*=10
- cur 如果为 0，那么与low无关，接下来会降为9...1，一共high轮，所以有 high*digit 个1
- cur 如果为 1，除了降为0之后的情况外，low位减少都要带着这个1，要多 low+1 个1
- cur 如果大于1，除了降为0之后的情况外，它降为1时low位要经历完整的最大到最小，所以要多 low*digit 个1

```python
def countDigitOne(self, n: int) -> int:
    digit = 1
    high, cur, low = n // 10, n % 10, 0
    res = 0
    while high != 0 or cur != 0:
        res += high * digit
        if cur == 1:
            res += low + 1
        elif cur != 0:
            res += digit
        low += cur * digit
        cur = high % 10
        high //= 10
        digit *= 10
    return res
```

### 44-数字序列中的某一位

```python
'''
    数字范围    数量   位数    占多少位
    1-9        9      1       9
    10-99      90     2       180
    100-999    900    3       2700
    1000-9999  9000   4       36000
'''
def findNthDigit(self, n: int) -> int:
    start = 1
    count = 9
    digit = 1
    while n > count:
        n -= count
        digit += 1
        start *= 10
        count = digit * start * 9
    num = start + (n-1) // digit
    return int(str(num)[(n-1)%digit])
```

### 45-把数组排成最小数

- 转为字符串，按 a+b < b+a 排序
- python快排，不用设哨兵

```python
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
        if tmp > res:
            res = tmp
    return res
```

### 49-丑数

- i 指向的数只能乘2，j 只能乘3，k 只能乘 5
- 每次取三个数乘各自的因子后的最小值加入dp，同时每个乘积为最新dp的指针都 +1（去重）

```python
def nthUglyNumber(self, n: int) -> int:
        dp = [0] * n
        dp[0] = 1
        i = j = k = 0
        for x in range(1, n):
            u2, u3, u5 = dp[i]*2, dp[j]*3, dp[k]*5
            dp[x] = min(u2, u3, u5)
            if dp[x] == u2:
                i += 1
            if dp[x] == u3:
                j += 1
            if dp[x] == u5:
                k += 1
        return dp[-1]
```

### 53-2 0-n-1中缺失的数字

有序就想二分

### 56-1-数组中数字出现的次数

- 全员异或得到出现一次的两个数的异或值（出现两次的异或后抵消为0）
- 从低到高找到为1的位
- 用这一位区分，这一位为1的放一起，不为1的放一起，分别异或得到两个出现一次的数

```python
def singleNumbers(self, nums: List[int]) -> List[int]:
    tmp = 0
    for num in nums:
        tmp = tmp ^ num
    div = 1
    while div & tmp == 0:
        div = div << 1
    a, b = 0, 0
    for num in nums:
        if num & div:
            a = a ^ num
        else:
            b = b ^ num
    return [a, b]
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
- j 有 1到6 6中可能的点数，如果是1，那么总点数为 j 说明前一个是投出 j-1 的状态
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

### 61-扑克牌中的顺子

除0之外，如果有重复的，直接pass，然后计算最大值和最小值之差是否小于5

### 62-圆圈中最后剩下的数字

约瑟夫环问题：

- 假设我们知道了有 10 个人时，最终的胜利者位置为 i，那么下一轮 9 个人时，删掉一个人之后，下一个人成为队头，相当于所有人都往前移了 m 位（不是1位，因为起始位置变了），那么最终胜利者位置也前移了 m 位 i-m
- 那么反推，如果9个人时最终胜利者位置为 i，那么10个人时位置就后移m位 i+m，考虑越界，要 % 这一轮的人数 10
- 1个人时，胜利者位置为 0

```python
def lastRemaining(self, n: int, m: int) -> int:
    index = 0
    for i in range(2, n+1):
        index = (index + m) % i
    return index
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

![Picture1.png](https://cdn.jsdelivr.net/gh/Hui4401/imgbed/img/2021/03/20/20210320105303.png)

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



