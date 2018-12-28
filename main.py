from learner import Learner
from word2vec import Word2VecBuilder

w2v = Word2VecBuilder('train_test.txt', 'word2vec.model', 50, 5, 5, 4, 1)

while True:
    learner = Learner('word2vec.model')
    learner.get_input('train.txt')
    #learner.split_train_test(90)
    learner.get_test_input('test.txt')
    learner.learn(0.9)
    learner.test_classifier()
    learner.training_accuracy(8)