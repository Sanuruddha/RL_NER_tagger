from learner import Learner
from word2vec import Word2VecBuilder
from preprocessor import Preprocessor

w2v = Word2VecBuilder('train_test.txt', 'word2vec.model', 50, 5, 5, 4, 1)
pre_proc = Preprocessor()

sets = {0: 100, 1: 250, 2: 500, 3: 1000, 4: 2000, 5: 3000, 6: 4000,  7: 5000, 8: 6000, 9: 7452}

train_set = pre_proc.get_input('train.txt')
test_set = pre_proc.get_input('test.txt')
for i in range(10):
    test_results = []
    train_results = []
    f_test = open(str(sets[i])+'test', 'w')
    f_train = open(str(sets[i])+'train', 'w')
    split_train_set = [train_set[0][:sets[i]], train_set[1][:sets[i]]]
    for j in range(10):
        learner = Learner('word2vec.model', split_train_set, test_set)
        learner.learn(0.9)
        testing_acc = learner.test_classifier()
        training_acc = learner.training_accuracy(8)
        test_results.append(str(testing_acc * 100))
        train_results.append(str(training_acc * 100))
    for k in range(len(test_results)):
        f_test.write(test_results[k] + " ")
        f_train.write(train_results[k] + " ")
    f_test.close()
    f_train.close()



