import numpy as np


def one_hot( x):
    z = np.zeros(shape=[4, 10])
    for i in range(4):
        index = int(x[i])
        z[i][index] += 1
    return z

a = '1234'
print(one_hot(a))