from model import Learner
from word2vec import Word2Vec

w2v = Word2Vec()
w2v = w2v.create('train_test.txt', 'word2vec.model')   # Word2Vec.create(training_corpus, output_filename, size=50, window=5,
                                                       #                min_count=5, workers=4, sg=1)

while True:
    learner = Learner('word2vec.model')   #Learner takes a word2vec embedding model as input
    learner.get_input('train.txt')
    #learner.split_train_test(90)
    learner.get_test_input('test.txt')
    learner.learn(0.9)
    learner.test_classifier()
    learner.training_accuracy(8)