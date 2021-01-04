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

n_points_per_cluster_total = 2000000
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
print('X labels_true Len', len(labels_true))
print('X labels_true ', labels_true)

# set1
X1 = X[:, 0:25]
#print('X1', X1.shape)
#print(X1)

X2 = X[:, 25:50]
X3 = X[:, 50:75]
X4 = X[:, 75:100]
print(X4.shape)
print('###############################################')
###############
from sklearn.cluster import KMeans
# Try X1 with kmeans
kmeans = KMeans(n_clusters=100, random_state=0).fit(X1)
centers = kmeans.cluster_centers_
print('centers', centers)
print('kmeans.labels_', kmeans.labels_[:10])
print('kmeans.labels_ len', len(kmeans.labels_))
print('###############################################')

# try X2 with dbscan
from sklearn.cluster import DBSCAN
clustering = DBSCAN(eps=15, min_samples=2, n_jobs=-1).fit(X2)

print('dbscan clustering.labels_', clustering.labels_[50:60])
print('dbscan clustering.labels_ len', len(clustering.labels_))
labels = clustering.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('X1 : Number of cluster in DBSCANN :  % d ' % n_clusters_)
# print('X1: Elapsed time to cluster in DBSCANN :  %.4f s ' % db_time_dbscan_process)
print('###############################################')

# Optics with X3
from sklearn.cluster import OPTICS
optics = OPTICS(min_samples=10, xi=.05, min_cluster_size=.05, n_jobs=-1).fit(X3)
print('Optics clustering labels', optics.labels_)
print(' Len Optics clustering labels', len(optics.labels_))
op_labels = optics.labels_
print('Optics labels :', op_labels)
n_clusters_op_ = len(set(op_labels)) - (1 if -1 in op_labels else 0)
print('Number CLuster of Optics is ', n_clusters_op_)
print('optics The cluster ordered list of sample indices. ', optics.ordering_)
print('###############################################')

from sklearn.cluster import MiniBatchKMeans
db_time_minibatch = time.time()
_MiniBatchKMeans = MiniBatchKMeans(n_clusters=100,random_state = 0,batch_size = 300, max_iter = 100).fit(X4)

db_time_process = time.time() - db_time_minibatch
print('Elapsed time to cluster in MInibatch kmeans :  %.4f s ' % db_time_process)
cluster_MiniBatchKMeans = _MiniBatchKMeans.cluster_centers_
labels_MiniBatchKMeans = _MiniBatchKMeans.labels_
print('labels_MiniBatchKMeans',labels_MiniBatchKMeans)
print('###############################################')
total_labels = kmeans.labels_ + clustering.labels_ + op_labels + labels_MiniBatchKMeans
print('total_labels',total_labels)
# find same element in list
import collections
same_element = [item for item, count in collections.Counter(total_labels).items() if count > 1]
differences_element = [item for item, count in collections.Counter(total_labels).items() if count == 1]
print('same_element',same_element)
print('differences_element',differences_element)
print('len of total labels in init vector', len(same_element))
# len of total labels in init vector 77 -> 77 cluster or what happend ?
