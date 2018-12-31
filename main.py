from learner import Learner
from word2vec import Word2VecBuilder
from preprocessor import Preprocessor
import matplotlib.pyplot as plt


w2v = Word2VecBuilder('train_test.txt', 'word2vec.model', 50, 5, 5, 4, 1)
pre_proc = Preprocessor()

test_results = []
train_results = []

for i in range(10):
    learner = Learner('word2vec.model', pre_proc.get_input('train.txt'), pre_proc.get_input('test.txt'))
    learner.learn(0.9)
    testing_acc = learner.test_classifier()
    training_acc = learner.training_accuracy(8)
    test_results.append(testing_acc * 100)
    train_results.append(training_acc * 100)


# plt.plot([i+1 for i in range(10)], test_results, linestyle='-', marker='o')
# plt.axis([0, 11, -1, 1])
# plt.show()
#

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter([i+1 for i in range(10)], test_results,  s=10, c='b', marker="s", label='test')
ax1.plot([i+1 for i in range(10)], test_results)
ax1.axis([0, 11, 0, 100])
ax1.scatter([i+1 for i in range(10)], train_results, s=10, c='r', marker="o", label='train')
ax1.plot([i+1 for i in range(10)], train_results)
ax1.axis([0, 11, 0, 100])
plt.legend(loc='upper left')
plt.show()