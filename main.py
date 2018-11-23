from model import Learner

while True:
    learner = Learner()
    learner.get_input('train.txt')
    #learner.split_train_test(90)
    learner.get_test_input('test.txt')
    learner.learn(0.9)
    learner.test_classifier()
    learner.training_accuracy(8)