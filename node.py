# tested
class Node:
    def __init__(self, sentence, labels):
        self._sentence = sentence
        self._labels = labels
        self._cost = 0
        self._optimal_action = None

    def get_sentence(self):
        return self._sentence

    def get_labels(self):
        return self._labels

    def set_labels(self, labels):
        self._labels = labels

    def set_optimal_action(self, action):
        self._optimal_action = action

    def get_optimal_action(self):
        return self._optimal_action

    def set_cost(self, cost):
        self._cost = cost

    def get_cost(self):
        return self._cost