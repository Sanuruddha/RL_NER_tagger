from induced_tree import InducedTree
import tensorflow as tf
from gensim.models import Word2Vec
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Learner:  # LOLS algorithm is implemented here
    def __init__(self,
                 length=10,
                 learning_rate=0.001,
                 n_input=90,
                 n_classes=2,
                 n_hidden_1=10,
                 n_hidden_2=10):
        self._x = None
        self._y = None
        self._induced_tree = None
        self.input = []
        self.labels = []
        self.training_data = []
        self.training_labels = []
        self.testing_data = []
        self.testing_labels = []
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
        self._actions = {'NNE': 0, 'NE': 1}
        self._init_classifier()

    def get_input(self, fname):
        import codecs
        sentences = [[]]
        labels = [[]]
        with codecs.open(fname, 'rb', encoding='utf-16', errors='ignore') as infile:
            lines = infile.readlines()
            EOF = False
            count = 0
            for line in lines:
                try:
                    word, label = line.split()
                    sentences[len(sentences) - 1].append(word)
                    if label == 'O':
                        label = 0
                    else:
                        label = 1
                    labels[len(labels) - 1].append(label)
                    count += 1
                    if count == 10:
                        count = 0
                        sentences.append([])
                        labels.append([])
                except:
                    continue

        self.input = sentences
        self.labels = labels

    def print_input_head(self):
        for i in self.input:
            print(i[0]),
            print(i[1])

    def split_train_test(self, percentage):
        length = len(self.input)
        train_length = (length * percentage) / 100

        self.training_data = self.input[:train_length]
        self.training_labels = self.labels[:train_length]
        self.testing_data = self.input[train_length:]
        self.testing_labels = self.labels[train_length:]
        print(len(self.training_data))
        print(len(self.testing_data))

    def learn(self, beta):
        import numpy as np
        training_data = self.training_data
        training_labels = self.training_labels
        actions = self._actions
        #for i in range(len(training_data)):
        for i in range(100):
            print("example " + str(i))
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
                    if action_cost < min_cost:
                        min_cost = action_cost
                    costs.append(action_cost)
                for cost_index in range(len(costs)):
                    costs[cost_index] = costs[cost_index] - min_cost
                feature_vector = self._generate_feature_vector(s)
                experience.append([feature_vector, costs])
                s = learned_policy[s]


            experience = self._make_X_Y(experience)

            self._train_classifier(experience)

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

        # Initializing the variables
        init = tf.global_variables_initializer()
        # Launch the graph
        self._tf_session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self._tf_session.run(init)

    def _train_classifier(self, experience):
        inp = experience[0]
        inp = np.array([np.array(xi) for xi in inp])
        #inp = np.array(inp, dtype=np.float64)
        labels = experience[1]
        labels = np.array([np.array(xi) for xi in labels])
        #labels = np.array(labels, dtype=np.float64)

        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = self._tf_session.run([self._optimizer, self._cost], feed_dict={self._x: inp,
                                                                              self._y: labels})
        print("cost=", "{:.9f}".format(c))
        # Display logs per epoch step
        print("Optimization Finished!")

    def test_classifier(self):
        X_test = self.testing_data
        Y_test = self.testing_labels
        accuracy = 0.
        for i in range(10):
            prediction = self.predict_labels(X_test[i])
            correct_count = 0.
            for j in range(len(prediction)):
                if prediction[j] == Y_test[i][j]:
                    correct_count += 1
            accuracy += (correct_count/len(prediction))
        accuracy = accuracy/100
        print("accuracy", accuracy)

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
        init = tf.global_variables_initializer()
        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            weights = self._weights_biases['weights']
            biases = self._weights_biases['biases']
            self._pred = self._multilayer_perceptron(tf, weights, biases)
            pred = tf.nn.softmax(self._pred)
            prediction = tf.argmin(pred, 1)
            feed_dict = {self._x: [inp]}
            action = sess.run(prediction, feed_dict)
            return action

    def _roll_out(self, policy, policy_type, state, action, induced_tree, labels):
        correct_count = 0.
        if policy_type is 0:
            state_n = induced_tree.transition(state, action)
            node_n = induced_tree.get_node(state_n)
            partial_labels = node_n.get_labels()
            #print("labels ", labels)
            #print("partial labels", partial_labels)
            for i in range(len(partial_labels)):
                if partial_labels[i] == labels[i]:
                    correct_count += 1

        elif policy_type is 1:
            state_n = induced_tree.transition(state, action)
            while state_n in policy:
                state_n = policy[state_n]

            node = induced_tree.get_node(state_n)
            prediction = node.get_labels()
            #print("labels ", labels)
            #print("prediction", prediction)
            for ind in range(len(labels)):
                if labels[ind] == prediction[ind]:
                    correct_count += 1
        cost = (1-correct_count/len(labels))
        return cost

    def _induce_tree(self, sentence):
        return InducedTree(sentence)

    def _generate_reference_policy(self, induced_tree, labels):
        self._generate_node_costs(induced_tree, labels)
        policy = self._get_reference_policy(induced_tree)
        return policy

    def _generate_node_costs(self, tree, labels):
        tree_dict = tree.get_dictionary()
        for i in tree_dict:
            node = tree_dict[i]
            partial_labels = node.get_labels()
            correct_count = 0
            num_of_labels = len(labels)
            none_index = -1
            for j in range(len(partial_labels)):
                if partial_labels[j] is None:
                    none_index = j
                    break
                elif partial_labels[j] == labels[j]:
                    correct_count += 1
            cost = 1-(correct_count/num_of_labels)
            node.set_cost(cost)
            node.set_optimal_action(labels[none_index])

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

        word2vec_model = Word2Vec.load("word2vec.model")
        partial_labels = self._induced_tree.get_node(state).get_labels()
        sentence = self._induced_tree.get_node(state).get_sentence()
        word_index = -1
        for i in range(len(partial_labels)):
            if partial_labels[i] is None:
                word_index = i
                break
        previous_word_vector = [0. for i in range(30)]
        next_word_vector = [0. for i in range(30)]
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
            return [0. for i in range(30)]