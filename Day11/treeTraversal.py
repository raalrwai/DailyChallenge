# tree_algorithms_demo.py

from collections import deque

# ---------------------------
# Tree Node Definition
# ---------------------------
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# ---------------------------
# Sample Tree Construction
# ---------------------------
# Tree Structure:
#         1
#       /   \
#      2     3
#     / \     \
#    4   5     6
root = TreeNode(1)
root.left = TreeNode(2, TreeNode(4), TreeNode(5))
root.right = TreeNode(3, None, TreeNode(6))

# ---------------------------
# 1. BFS Traversal (Level Order)
# ---------------------------
# Returns a list of values in level-order
def bfs_traversal(root):
    if not root:
        return []
    queue = deque([root])
    result = []
    while queue:
        node = queue.popleft()
        result.append(node.val)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return result

# ---------------------------
# 2. Preorder Traversal
# ---------------------------
# Returns values in root -> left -> right order
def preorder(root):
    if not root:
        return []
    return [root.val] + preorder(root.left) + preorder(root.right)

# ---------------------------
# 3. Inorder Traversal
# ---------------------------
# Returns values in left -> root -> right order
def inorder(root):
    if not root:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)

# ---------------------------
# 4. Postorder Traversal
# ---------------------------
# Returns values in left -> right -> root order
def postorder(root):
    if not root:
        return []
    return postorder(root.left) + postorder(root.right) + [root.val]

# ---------------------------
# 5. Maximum Depth of Tree
# ---------------------------
# Returns the maximum depth (height) of the tree
def max_depth(root):
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))

# ---------------------------
# 6. Minimum Depth of Tree
# ---------------------------
# Returns the minimum depth from root to nearest leaf
def min_depth(root):
    if not root:
        return 0
    if not root.left or not root.right:
        return 1 + max(min_depth(root.left), min_depth(root.right))
    return 1 + min(min_depth(root.left), min_depth(root.right))

# ---------------------------
# 7. Level Sums
# ---------------------------
# Returns a list of sums for each level in the tree
def level_sums(root):
    if not root:
        return []
    queue = deque([root])
    result = []
    while queue:
        level_size = len(queue)
        level_sum = 0
        for _ in range(level_size):
            node = queue.popleft()
            level_sum += node.val
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level_sum)
    return result

# ---------------------------
# 8. Maximum Level Sum
# ---------------------------
# Returns the level number with the largest sum of values
def max_level_sum(root):
    if not root:
        return 0
    queue = deque([root])
    level = 1
    max_sum = float('-inf')
    best_level = 1
    while queue:
        level_size = len(queue)
        level_sum = 0
        for _ in range(level_size):
            node = queue.popleft()
            level_sum += node.val
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        if level_sum > max_sum:
            max_sum = level_sum
            best_level = level
        level += 1
    return best_level

# ---------------------------
# 9. Count Total Nodes
# ---------------------------
# Returns the total number of nodes in the tree
def count_nodes(root):
    if not root:
        return 0
    return 1 + count_nodes(root.left) + count_nodes(root.right)

# ---------------------------
# 10. Check if Tree is Balanced
# ---------------------------
# Returns True if the tree is height-balanced, else False
def is_balanced(root):
    def helper(node):
        if not node:
            return 0, True
        left_height, left_balanced = helper(node.left)
        right_height, right_balanced = helper(node.right)
        balanced = left_balanced and right_balanced and abs(left_height - right_height) <= 1
        return 1 + max(left_height, right_height), balanced
    return helper(root)[1]

# ---------------------------
# Testing / Demo
# ---------------------------
if __name__ == "__main__":
    print("BFS Traversal:", bfs_traversal(root))
    print("Preorder Traversal:", preorder(root))
    print("Inorder Traversal:", inorder(root))
    print("Postorder Traversal:", postorder(root))
    print("Max Depth:", max_depth(root))
    print("Min Depth:", min_depth(root))
    print("Level Sums:", level_sums(root))
    print("Max Level Sum:", max_level_sum(root))
    print("Total Nodes:", count_nodes(root))
    print("Is Balanced:", is_balanced(root))
