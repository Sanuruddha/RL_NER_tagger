from model import Learner

learner = Learner()
learner.get_input('train.txt')
#learner.split_train_test(90)
learner.get_input('test.txt')
learner.learn(0.9)
learner.test_classifier()