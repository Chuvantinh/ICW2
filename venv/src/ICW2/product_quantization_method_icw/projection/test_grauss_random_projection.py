import numpy as np
from sklearn import random_projection
X = np.random.rand(100, 10000)
transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(X)

print(X.shape)
print('X',X)
print(X_new.shape)
print('X_new', X_new)
#(100, 3947)