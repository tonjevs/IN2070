import numpy as np

A = np.array([[221,396,(221*396),1], [221,397,(221*397),1], [222,396,(222*396),1], [222,397,(222*397),1]])
B = np.array([18,45,52,36])
X = np.linalg.solve(A,B)
y = X[0]*221.3 + X[1]*396.7 + X[2]*396.7*221.3 + X[3]
print(y)
