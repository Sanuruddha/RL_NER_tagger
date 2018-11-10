import numpy as np

choice = np.random.choice(
  ['pooh', 'rabbit', 'piglet', 'Christopher'],
  1,
  p=[0.5, 0.1, 0.1, 0.3]
)

pooh = 0
rabbit = 0
piglet = 0
christopher = 0

for i in range(10):
    choice = np.random.choice(['pooh', 'rabbit', 'piglet', 'Christopher'],
                              1,
                              p=[0.6, 0.1, 0.1, 0.2])
    if choice == 'pooh':
        pooh += 1
    elif choice == 'rabbit':
        rabbit += 1
    elif choice == 'piglet':
        piglet += 1
    elif choice == 'Christopher':
        christopher += 1

print("pooh:", pooh, "rabbit:", rabbit, "piglet:", piglet, "christopher", christopher)

class TreeNode:
    def __init__(self,data,labels):
        self.data = data
        self.labels = labels
        self.parent = None
        self.right_child = None
        self.left_child = None
        self.number=None


class InducedTree:
    def __init__(self,n):
        self.root = None
        self.list_length = n
        self.size = 0

    def induce_tree(self, word_list):
        if not len(word_list) == self.list_length:
            print("List length does not match the tree size")
            return
        node = TreeNode(word_list, [0 for i in range(self.list_length)])
        self.root = node
        self.size = 1
        self.root.number = self.size
        num = (2 ** (self.list_length + 1)) - 2
        print(num)
        for i in range(1,num):
            node = TreeNode(word_list, [0 for j in range(self.list_length)])
            self.add_node(node)

    def add_node(self, node):
        root = self.root
        queue = [root]
        self.size+=1
        node.number = self.size
        while not len(queue) == 0:
            current=queue[0]
            del queue[0]
            if current.left_child is None:
                current.left_child = node
                return
            elif current.right_child is None:
                current.right_child = node
                return
            else:
                queue.append(current.left_child)
                queue.append(current.right_child)
                continue

    def search(self, state_number):
        root = self.root
        self._search_recursive(root, state_number)

    def _search_recursive(self,node, state_number):
        if node.number == state_number:
            return node
        self._search_recursive(node.left_child, state_number)
        self._search_recursive(node.right_child, state_number)




    def print_tree(self):
        self.printrec(self.root)

    def print_rec(self, node):
        print(node.number)
        if node.left_child is not None:
            self.print_rec(node.left_child)
        if node.right_child is not None:
            self.print_rec(node.right_child)

    def print_like_tree(self):
        queue = [self.root]

        while not len(queue) == 0:
            spaces = (tree.size / 2) - len(queue)
            print("           " * (spaces - 1)),
            for i in range(len(queue)):
                print(queue[i].labels),
            print("\n\n\n")
            new_queue = []
            for i in range(len(queue)):
                if not queue[i].left_child == None:
                    new_queue.append(queue[i].left_child)
                if not queue[i].right_child == None:
                    new_queue.append(queue[i].right_child)
            queue = new_queue


tree = InducedTree(7)

tree.induce_tree(['HI', 'you', 'hey', 'yo', 'dude', 'darl', 'sup'])

