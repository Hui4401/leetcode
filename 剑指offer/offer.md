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
    res = 1
    # 分出尽可能多的3，4比较特殊分为2*2更好，所以剩余<=4时就直接将剩下的相乘返回
    while n > 4:
        res *= 3
        n -= 3
    return res * n
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
    m = len(matrix)
    if m == 0:
        return []
    n = len(matrix[0])
    if n == 0:
        return []
    res = []
    # 行首尾，列首尾
    up, down, left, right = 0, m-1, 0, n-1
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

### 44-数字序列中的某一位

### 53-2 0-n-1中缺失的数字

有序就想二分

### 59-1-滑动窗口最大值

单调队列，每次右移，如果移出去的元素等于队列头，从队列弹出这个元素，同时从尾部弹出所有小于窗口新增元素的元素，然后把新增元素加入队列

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

