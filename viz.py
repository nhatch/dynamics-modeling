#!/home/nhatch2/miniconda3/bin/python3

import numpy as np
import matplotlib.pyplot as plt

target = 'sim_data'
X= np.loadtxt(target + '/np.txt')
N = X.shape[0]
D = 2
train_N = N // 5 * 4
test_N = N - train_N
x_train = X[:train_N,:D]
y_train = X[:train_N,D:]
x_test =  X[train_N:,:D]
y_test =  X[train_N:,D:]

px = X[:,0]
py = X[:,2]

#px = X[:,1]
#py = X[:,5]

plt.scatter(px,py)
plt.show()

