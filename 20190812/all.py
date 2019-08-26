import numpy as np
a = np.array([[1,2,3,4],[1,2,3,4]])
b = np.array([[1,2,3,4],[1,2,2,4]])
t = np.mean(np.all(a==b,axis=1))
print(t)