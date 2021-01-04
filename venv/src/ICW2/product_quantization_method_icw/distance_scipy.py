# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.datasets.samples_generator import make_blobs

n_points_per_cluster_total = 1000
size_colum = 100
centers = np.random.randint(-30, 30, size=(size_colum,size_colum))
#print('centers', centers[0])
# train
X1, labels_true = make_blobs(n_samples=n_points_per_cluster_total,
                            centers=centers,
                            n_features=size_colum,
                            cluster_std=0.4,
                            random_state=0)
print(X1[1])
print(labels_true)
print(len(labels_true))
distance = pdist(X1, 'euclidean')
print('distance',distance)
