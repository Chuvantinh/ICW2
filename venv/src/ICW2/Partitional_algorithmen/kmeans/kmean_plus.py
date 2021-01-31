from sklearn.cluster import kmeans_plusplus
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import time
# Generate sample data
n_points_of_cluster = 4000000
size_colum = 100
centers = np.random.randint(-20, 20, size=(size_colum,size_colum))

X, labels_true = make_blobs(n_samples=n_points_of_cluster,centers=centers, n_features=size_colum, cluster_std=0.4, random_state=0)

# Calculate seeds from kmeans++
db_time = time.time()
centers_init, indices = kmeans_plusplus(X, n_clusters=4,
                                        random_state=0)

db_time_process = time.time() - db_time
print('Elapsed time to cluster in kmeans :  %.4f s ' % centers_init)