from induced_tree import InducedTree
import tensorflow as tf
from gensim.models import Word2Vec
import numpy as np
import os
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Learner:  # LOLS algorithm is implemented here
    def __init__(self,
                 vw_model,
                 training_set,
                 testing_set,
                 length=10,
                 learning_rate=0.001,
                 n_input=150,
                 n_classes=2,
                 n_hidden_1=10,
                 n_hidden_2=10):
        self._model = vw_model
        self._x = None
        self._y = None
        self._induced_tree = None
        self.input = []
        self.labels = []
        self.training_data = training_set[0]
        self.training_labels = training_set[1]
        self.testing_data = testing_set[0]
        self.testing_labels = testing_set[1]
        self.classifier = None
        self.sentence_length = length
        self._weights_biases ={}
        self._learning_rate = learning_rate
        self._n_input = n_input
        self._n_classes = n_classes
        self._n_hidden_1 = n_hidden_1
        self._n_hidden_2 = n_hidden_2
        self._pred = None
        self._tf_session = None
        self._cost = None
        self._optimizer = None
        self._prediction = None
        self._actions = {'NNE': 0, 'NE': 1}
        self._experiences = []
        self._init_classifier()


    def learn(self, beta):
        import numpy as np
        training_data = self.training_data
        training_labels = self.training_labels
        actions = self._actions
        for i in range(len(training_data)):
        #for i in range(2):
            #print("example " + str(i))
            self._induced_tree = self._induce_tree(training_data[i])
            learned_policy = self._generate_learned_policy()
            reference_policy = self._generate_reference_policy(self._induced_tree, training_labels[i])
            experience = []
            s = 0
            for j in range(len(training_data[i])):
                costs = []
                min_cost = 1
                for action in actions:
                    choice = int([np.random.choice([0, 1],
                                                   1,
                                                   p=[beta, 1 - beta])][0][0])
                    if choice == 0:
                        pol = reference_policy
                    else:
                        pol = learned_policy
                    action_cost = self._roll_out(pol, choice, s, actions[action], self._induced_tree, training_labels[i])
                    costs.append(action_cost)

                feature_vector = self._generate_feature_vector(s)
                experience.append([feature_vector, costs])
                s = learned_policy[s]

            self._experiences += experience
            if len(self._experiences) < 40:
                sample = self._make_X_Y(self._experiences)
            else:
                sample = self._sample(self._experiences)
            self._train_classifier(sample, i)

    def _sample(self, experiences):
        X = []
        Y = []
        sample = random.sample(experiences, 40)
        for i in range(len(sample)):
            X.append(sample[i][0])
            Y.append(sample[i][1])
        return [X, Y]

    def _make_X_Y(self, experience):
        X = []
        Y = []
        for i in range(len(experience)):
            X.append(experience[i][0])
            Y.append(experience[i][1])

        return [X, Y]

    def _multilayer_perceptron(self, tf, weights, biases):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(self._x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        return out_layer

    def _init_classifier(self):

        # tf Graph input
        self._x = tf.placeholder("float", [None, self._n_input])
        self._y = tf.placeholder("float", [None, self._n_classes])

        # Store layers weight & bias
        weights = {
            'h1': tf.Variable(tf.random_normal([self._n_input, self._n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([self._n_hidden_1, self._n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self._n_hidden_2, self._n_classes]))
        }

        biases = {
            'b1': tf.Variable(tf.random_normal([self._n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([self._n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self._n_classes]))
        }

        self._weights_biases['weights'] = weights
        self._weights_biases['biases'] = biases

        # Construct model
        self._pred = self._multilayer_perceptron(tf, weights, biases)

        # Define loss and optimizer
        self._cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self._pred, labels=self._y))
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(self._cost)
        pred = tf.nn.softmax(self._pred)
        self._prediction = tf.argmin(pred, 1)

        # Initializing the variables
        init = tf.global_variables_initializer()
        # Launch the graph
        self._tf_session = tf.Session(config=tf.ConfigProto())
        self._tf_session.run(init)

    def _train_classifier(self, experience, counter):
        inp = experience[0]
        inp = np.array([np.array(xi) for xi in inp])
        labels = experience[1]
        labels = np.array([np.array(xi) for xi in labels])

        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = self._tf_session.run([self._optimizer, self._cost], feed_dict={self._x: inp,
                                                                              self._y: labels})
        if counter % 200 == 0:
            print("cost=", "{:.9f}".format(c))

    def test_classifier(self):
        X_test = self.testing_data
        print("test size", len(X_test))
        Y_test = self.testing_labels
        accuracy = 0.
        for i in range(len(X_test)):
        #for i in range(1):
            prediction = self.predict_labels(X_test[i])
            correct_count = 0.
            for j in range(len(prediction)):
                if prediction[j] == Y_test[i][j]:
                    correct_count += 1
            accuracy += (correct_count/len(prediction))
        accuracy = accuracy/len(X_test)
        return accuracy

    def training_accuracy(self, percentage):
        size = (len(self.training_data) * percentage) / 100
        print("train size", size)
        start_index = random.randint(0, len(self.training_data) - size - 1)
        print("start", start_index)
        X_test = self.training_data[start_index: start_index+size]
        Y_test = self.training_labels[start_index: start_index+size]
        accuracy = 0.
        for i in range(len(X_test)):
        #for i in range(1):
            prediction = self.predict_labels(X_test[i])
            correct_count = 0.
            for j in range(len(prediction)):
                if prediction[j] == Y_test[i][j]:
                    correct_count += 1
            accuracy += (correct_count / len(prediction))
        accuracy = accuracy / len(X_test)
        return accuracy

    def predict_labels(self, sentence):
        self._induced_tree = self._induce_tree(sentence)
        learned_policy = self._generate_learned_policy()
        state = 0
        prediction = []
        while state in learned_policy:
            if self._induced_tree.transition(state, 0) == learned_policy[state]:
                prediction.append(0)
            else:
                prediction.append(1)
            state = learned_policy[state]

        return prediction

    def predict(self, inp):
        feed_dict = {self._x: [inp]}
        action = self._tf_session.run(self._prediction, feed_dict)
        return action

    def _roll_out(self, policy, policy_type, state, action, induced_tree, labels):
        node = induced_tree.get_node(state)
        cost = node.get_action_cost(action)

        return cost

    def _induce_tree(self, sentence):
        return InducedTree(sentence)

    def _generate_reference_policy(self, induced_tree, labels):
        self._generate_node_costs(induced_tree, labels)
        policy = self._get_reference_policy(induced_tree)
        return policy

    def _generate_node_costs(self, tree, labels):
        current = 0
        self._recurse(current, tree, labels)

    def _recurse(self, state, tree, labels):
        if tree.is_leaf(state):
            node = tree.get_node(state)
            partial_labels = node.get_labels()
            correct_count = 0
            for j in range(len(partial_labels)):
                if partial_labels[j] == labels[j]:
                    correct_count += 1
            cost = 1 - (float(correct_count) / len(labels))
            node.set_cost(cost)
            return len(labels) - correct_count
        else:
            node = tree.get_node(state)
            left = self._recurse(tree.get_left_child(state), tree, labels)
            right = self._recurse(tree.get_right_child(state), tree, labels)
            total = float(left + right)
            node.set_action_cost(0, left / total)
            node.set_action_cost(1, right / total)
            return total

    def _get_reference_policy(self, tree):
        tree_dict = tree.get_dictionary()
        policy = {}
        for i in tree_dict:
            action = tree_dict[i].get_optimal_action()
            if action == 0:
                s = tree.get_left_child(i)
            else:
                s = tree.get_right_child(i)
            policy[i] = s
        return policy

    def _generate_learned_policy(self):

        learned_policy = {}
        tree = self._induced_tree
        current_state = 0
        while not tree.is_leaf(current_state):
            current_feature_vector = self._generate_feature_vector(current_state)
            action = self.predict(current_feature_vector)
            action = np.array(action).tolist()[0]
            if action is 0:
                learned_policy[current_state] = tree.get_left_child(current_state)
                current_state = tree.get_left_child(current_state)
            else:
                learned_policy[current_state] = tree.get_right_child(current_state)
                current_state = tree.get_right_child(current_state)
        return learned_policy

    def _generate_feature_vector(self, state):

        word2vec_model = Word2Vec.load(self._model)
        partial_labels = self._induced_tree.get_node(state).get_labels()
        sentence = self._induced_tree.get_node(state).get_sentence()
        word_index = -1
        for i in range(len(partial_labels)):
            if partial_labels[i] is None:
                word_index = i
                break
        previous_word_vector = [0. for i in range(50)]
        next_word_vector = [0. for i in range(50)]
        if word_index > 1:
            previous_word = sentence[word_index-1]
            previous_word_vector = self._get_word_embedding(word2vec_model.wv, previous_word)
            previous_word_vector = np.array(previous_word_vector).tolist()
        current_word = sentence[word_index]
        if word_index < len(sentence) - 1:
            next_word = sentence[word_index+1]
            next_word_vector = self._get_word_embedding(word2vec_model.wv, next_word)
            next_word_vector = np.array(next_word_vector).tolist()
        current_word_vector = self._get_word_embedding(word2vec_model.wv, current_word)

        current_word_vector = np.array(current_word_vector).tolist()
        vector = previous_word_vector + current_word_vector + next_word_vector
        return vector

    def _get_word_embedding(self, model, word):
        try:
            return model[word]
        except:
            return [0. for i in range(50)]