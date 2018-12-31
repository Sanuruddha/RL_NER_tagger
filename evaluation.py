import matplotlib.pyplot as plt
import numpy as np

sets = {0: 100,
        1: 250,
        # 2: 500,
        # 3: 1000,
        # 4: 2000,
        # 5: 3000,
        # 6: 4000,
        # 7: 5000,
        # 8: 6000,
        # 9: 7452
        }
test_results = {}
train_results = {}
for i in sets:
    test_results[sets[i]] = []
    train_results[sets[i]] = []

for i in sets:
    f_test = open(str(sets[i]) + 'test', 'r')
    f_train = open(str(sets[i]) + 'train', 'r')
    test_results[sets[i]] = f_test.readline().split()
    train_results[sets[i]] = f_train.readline().split()
    f_test.close()
    f_train.close()

colorArray = ['#FF6633', '#FFB399', '#FF33FF', '#FFFF99', '#00B3E6', '#E6B333',
              '#3366E6', '#999966', '#99FF99', '#B34D4D']
fig = plt.figure()
ax1 = fig.add_subplot(111)
print(len(test_results), len(train_results))
j = 0
for i in test_results:
    print([k+1 for k in range(len(test_results[i]))])
    print(test_results[i])
    ax1.scatter([k+1 for k in range(len(test_results[i]))], [float(l) for l in test_results[i]],  s=10, c=colorArray[j], marker="s", label=str(i))
    ax1.plot([k+1 for k in range(len(test_results[i]))], [float(l) for l in test_results[i]])
    # ax1.scatter([k for k in range(10)], train_results[i], s=10, c=colorArray[j], marker="o", label='train')
    # ax1.plot([k for k in range(10)], train_results[i])
    plt.setp(ax1.get_xticklabels(), visible=True)
    plt.xticks(np.arange(1, 11, 1))
    plt.yticks(np.arange(0, 105, 5))
    j += 1


ax1.set_xlabel('Number of training data', fontsize=12)
plt.legend(loc='upper left')
plt.show()