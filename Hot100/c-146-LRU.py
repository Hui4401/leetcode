# 先来一个双向链表
class Node:
    def __init__(self, key=0, value=0, pre=None, nex=None):
        self.key = key
        self.value = value
        self.pre = pre
        self.nex = nex
class List:
    def __init__(self):
        self.head = Node()
        self.tail = Node(pre=self.head)
        self.head.nex = self.tail
    def append(self, node):
        node.pre, node.nex = self.tail.pre, self.tail
        node.pre.nex = node.nex.pre = node
    def movetail(self, node):
        pre, nex = node.pre, node.nex
        pre.nex, nex.pre = nex, pre
        self.append(node)
    def pop(self):
        node = self.head.nex
        self.head.nex, node.nex.pre = node.nex, self.head
        return node.key

# 需要：1.hashmap来进行O1读写，2.双向链表来进行插入删除
# hashmap value存node节点，删除时要同时删除hashmap中的记录，反向操作，所以链表节点还要存key值
class LRUCache:

    def __init__(self, capacity: int):
        self.li = List()
        self.mp = {}
        self.maxl = capacity

    def get(self, key: int) -> int:
        node = self.mp.get(key, None)
        if node:
            self.li.movetail(node)
            return node.value
        return -1

    def put(self, key: int, value: int) -> None:
        # 能查到的话先更新值再移动到尾部
        if key in self.mp.keys():
            node = self.mp[key]
            node.value = value
            self.li.movetail(node)
            self.mp[key] = node
            return
        # 如果容量已经满了，删除最旧的节点
        if len(self.mp) >= self.maxl:
            del self.mp[self.li.pop()]
        node = Node(key, value)
        self.li.append(node)
        self.mp[key] = node


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)