import pqkmeans
import numpy as np
from sklearn.cluster import KMeans
# for data set
from sklearn.datasets.samples_generator import make_blobs
np.random.seed(42)

# time measure
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
time_all = time.time()
# Creating the original records in order to cluster it
n_points_per_cluster_total = 1000
size_colum = 100
centers = np.random.randint(-20, 20, size=(size_colum,size_colum))
#print('centers', centers[0])
# train
X, labels_true = make_blobs(n_samples=n_points_per_cluster_total,
                            centers=centers,
                            n_features=size_colum,
                            cluster_std=0.4,
                            random_state=0)
print('X shape', X.shape)
print(X)
#print('X labels_true Len', len(labels_true))
#print('X labels_true ', labels_true)
count = 1
for ( datapoint) in zip(X):
    count = count + 1
    if count < 20 :
        print(datapoint)