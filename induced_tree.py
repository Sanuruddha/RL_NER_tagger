from node import Node


class InducedTree:  # data structure that holds the induced tree
    def __init__(self, sentence):
        """
        dictionary = { 1: node, 2: node ....... }

        node = {_sentence : sentence,
                _labels : partial_labels,
                _cost: [ 0: left_cost, 1: right_cost],
                _optimal_action: 0 or 1
                }

        relations = { 1: [left child of 1, right child of 1], 2: [left child of 2, right child of 2]}

        """
        self._dictionary = {}
        self._relations = {}
        self._induce_tree(self._dictionary, self._relations, sentence)

    # tested
    def _induce_tree(self, dictionary, relations, sentence):
        length = len(sentence)

        for i in range(2 ** (length+1)):
            node = Node(sentence, [])
            dictionary[i] = node

        count = 1
        for i in range(2 ** length - 1):
            relations[i] = [count]
            count += 1
            relations[i].append(count)
            count += 1
        self._generate_labels(dictionary, length)

    # tested
    def _generate_labels(self, dictionary, levels):
        state = 0
        dictionary[0].set_labels([None] * levels)
        for level in range(1, levels + 1):
            num_of_nodes = 2 ** level
            num_of_nones = levels - level
            for node in range(num_of_nodes):
                state += 1
                labels = bin(node)[2:].zfill(level)
                labels = list(map(int, list(labels))) + num_of_nones * [None]
                dictionary[state].set_labels(labels)

    def get_right_child(self, state):
        if state in self._relations:
            return self._relations[state][1]
        return None

    def get_left_child(self, state):
        if state in self._relations:
            return self._relations[state][0]
        return None

    def get_node(self, state):
        if state in self._dictionary:
            return self._dictionary[state]
        else:
            return None

    def is_leaf(self, state):
        if self.get_left_child(state) is None and self.get_right_child(state) is None:
            return True
        else:
            return False

    def get_dictionary(self):
        return self._dictionary

    def transition(self, state, action):
        if self.is_leaf(state):
            return None
        elif action == 0:
            return self.get_left_child(state)
        elif action == 1:
            return self.get_right_child(state)
        else:
            raise ValueError("Invalid value for action")
