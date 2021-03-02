"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = []):
        self.val = val
        self.neighbors = neighbors
"""


class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        if not node: return
        lookup = {}
        def dfs(node):
            if node.val in lookup:
                return lookup[node.val]
            clone = Node(node.val)
            lookup[node.val] = clone
            for n in node.neighbors:
                clone.neighbors.append(dfs(n))
            
            return clone

        return dfs(node)
        