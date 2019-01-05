from learner import Learner
from word2vec import Word2VecBuilder
from preprocessor import Preprocessor

w2v = Word2VecBuilder('train_test.txt', 'word2vec.model', 7, 3, 1, 4, 1)
pre_proc = Preprocessor()

sets = {0: 100, 1: 250, 2: 500, 3: 1000, 4: 2000, 5: 3000, 6: 4000,  7: 5000, 8: 6000, 9: 7452}

train_set = pre_proc.get_input('train.txt')
test_set = pre_proc.get_input('test.txt')
for i in range(4, 10):
    test_accuracies = []
    train_accuracies = []
    test_precisions = []
    train_precisions = []
    test_recalls = []
    train_recalls = []
    f_test = open(str(sets[i])+'test', 'w')
    f_train = open(str(sets[i])+'train', 'w')
    split_train_set = [train_set[0][:sets[i]], train_set[1][:sets[i]]]
    for j in range(1):
        learner = Learner('word2vec.model', split_train_set, test_set, 10, 0.001, 21, 2, 6, 10)
        learner.learn(1)
        test_results = learner.test_classifier()
        train_results = learner.training_accuracy(sets[i]-10)
        test_accuracy = test_results[0]
        train_accuracy = train_results[0]
        test_precision = test_results[1]
        train_precision = train_results[1]
        test_recall = test_results[2]
        train_recall = train_results[2]

        test_accuracies.append(str(test_accuracy * 100))
        train_accuracies.append(str(train_accuracy * 100))
        test_precisions.append(str(test_precision * 100))
        train_precisions.append(str(train_precision * 100))
        test_recalls.append(str(test_recall * 100))
        train_recalls.append(str(train_recall * 100))
    for k in range(len(test_accuracies)):
        f_test.write(test_accuracies[k] + " ")
        f_train.write(train_accuracies[k] + " ")
    f_test.write("\n")
    f_train.write("\n ")
    for k in range(len(test_precisions)):
        f_test.write(test_precisions[k] + " ")
        f_train.write(train_precisions[k] + " ")
    f_test.write("\n")
    f_train.write("\n ")
    for k in range(len(test_recalls)):
        f_test.write(test_recalls[k] + " ")
        f_train.write(train_recalls[k] + " ")
    f_test.close()
    f_train.close()



