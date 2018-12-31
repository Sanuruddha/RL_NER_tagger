import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter([i+1 for i in range(10)], [89.567, 88.222, 91.344, 88.990, 87.534, 90.23, 92.6 ,89.0, 88.67, 91.345], s=10, c='r', marker="o", label='train')
ax1.plot([i+1 for i in range(10)], [89.567, 88.222, 91.344, 88.990, 87.534, 90.23, 92.6 ,89.0, 88.67, 91.345])
ax1.axis([0, 11, 0, 100])
plt.legend(loc='upper left')
plt.show()