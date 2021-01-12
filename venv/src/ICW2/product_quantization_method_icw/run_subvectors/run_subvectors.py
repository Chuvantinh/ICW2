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
n_points_per_cluster_total = 100000
size_colum = 100
centers = np.random.randint(-20, 20, size=(size_colum,size_colum))
#print('centers', centers[0])
# train
X, labels_true = make_blobs(n_samples=n_points_per_cluster_total,
                            centers=centers,
                            n_features=size_colum,
                            cluster_std=0.4,
                            random_state=0)
#print('X shape', X.shape)
#print('X labels_true Len', len(labels_true))
#print('X labels_true ', labels_true)

# set1
X1 = X[:, 0:25]
#print('X1', X1.shape)
#print(X1)

X2 = X[:, 25:50]
X3 = X[:, 50:75]
X4 = X[:, 75:100]
#print(X4.shape)
#print('###############################################')
###############
from sklearn.cluster import KMeans
# Try X1 with kmeans
kmeans = KMeans(n_clusters=50, random_state=0).fit(X1)
centers = kmeans.cluster_centers_
#print('centers', centers)
#print('kmeans.labels_', kmeans.labels_[:10])
#print('kmeans.labels_ len', len(kmeans.labels_))
print('###############################################')

# try X2 with dbscan
# from sklearn.cluster import DBSCAN
# clustering = DBSCAN(eps=15, min_samples=2, n_jobs=-1).fit(X2)
#
# print('dbscan clustering.labels_', clustering.labels_[50:60])
# print('dbscan clustering.labels_ len', len(clustering.labels_))
# labels = clustering.labels_
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# print('X1 : Number of cluster in DBSCANN :  % d ' % n_clusters_)
# print('X1: Elapsed time to cluster in DBSCANN :  %.4f s ' % db_time_dbscan_process)
kmeans2 = KMeans(n_clusters=50, random_state=0).fit(X2)
centers2 = kmeans2.cluster_centers_
# print('centers of kmeans 2', centers2)
# print('kmeans2.labels_', kmeans2.labels_[:10])
# print('kmeans2.labels_ len', len(kmeans2.labels_))
# print('###############################################')

# Try X3 with kmeans
kmeans3 = KMeans(n_clusters=50, random_state=0).fit(X3)
centers3 = kmeans3.cluster_centers_
# print('centers of kmeans 3', centers3)
# print('kmeans3.labels_', kmeans3.labels_[:10])
# print('kmeans3.labels_ len', len(kmeans3.labels_))
# print('###############################################')

print('###############################################')
kmeans4 = KMeans(n_clusters=50, random_state=0).fit(X4)
centers4 = kmeans4.cluster_centers_
# print('centers of kmeans 3', centers4)
# print('kmeans4.labels_', kmeans4.labels_[:10])
# print('kmeans4.labels_ len', len(kmeans4.labels_))
# print('###############################################')


# from sklearn.cluster import MiniBatchKMeans
# db_time_minibatch = time.time()
# _MiniBatchKMeans = MiniBatchKMeans(n_clusters=50,random_state = 0,batch_size = 300, max_iter = 100).fit(X4)

# db_time_process = time.time() - db_time_minibatch
# print('Elapsed time to cluster in MInibatch kmeans :  %.4f s ' % db_time_process)
# cluster_MiniBatchKMeans = _MiniBatchKMeans.cluster_centers_
# labels_MiniBatchKMeans = _MiniBatchKMeans.labels_
# print('labels_MiniBatchKMeans',labels_MiniBatchKMeans)
print('###############################################')
#total_labels = kmeans.labels_ + kmeans2.labels_ + kmeans3.labels_ + kmeans4.labels_
#total_labels = np.stack((kmeans.labels_, kmeans2.labels_, kmeans3.labels_, kmeans4.labels_), axis = 1)
total_labels = np.concatenate((kmeans.labels_, kmeans2.labels_, kmeans3.labels_, kmeans4.labels_), axis=None)
print(total_labels[20:30])
print('total_labels',total_labels.shape)
# filter unique index in array
Index_values = np.unique(total_labels, axis = 0)
print('index_values', Index_values)
print('len of index_values', Index_values.shape)
#import collections
#same_element = [item for item, count in collections.Counter(total_labels).items() if count > 1]
# differences_element = [item for item, count in collections.Counter(total_labels).items() if count == 1]

data_records = []
for index in Index_values:
    data_records = [datapoint for (i, datapoint) in zip(total_labels, X) if np.array_equal(i, index)]
    # for (i, datapoint) in zip(total_labels,X):
    #     if np.array_equal(i, index):
    #          data_records.append(datapoint)
    #         #data_records.push(datapoint)
data_records = np.array( data_records )
print('data_records', data_records)
print('len(data_records)', data_records.shape)
# last run with optics or dbscan to find the last cluster of the original set
# Optics with X
# from sklearn.cluster import OPTICS
# optics = OPTICS(min_samples=10, xi=.05, min_cluster_size=.05, n_jobs=-1).fit(data_records)
# print('Optics clustering labels', optics.labels_)
# print(' Len Optics clustering labels', len(optics.labels_))
# op_labels = optics.labels_
# print('Optics labels :', op_labels)
# n_clusters_op_ = len(set(op_labels)) - (1 if -1 in op_labels else 0)
# print('Number CLuster of Optics is ', n_clusters_op_)
# print('optics The cluster ordered list of sample indices. ', optics.ordering_)

print('###############################################')
time_kmean_last = time.time()
kmeans_last = KMeans(n_clusters=100, random_state=0).fit(data_records)
centers_last = kmeans_last.cluster_centers_
print('centers of kmeans_last', centers_last)
print('centers of kmeans_last', len(centers_last))
print('kmeans_last.labels_', kmeans_last.labels_[:10])
print('kmeans_last.labels_ len', len(kmeans_last.labels_))
print('###############################################')
time_last=  time.time() - time_kmean_last
time_total =  time.time() - time_all
print('time_last', time_last)
print('time_total', time_total)
# ergebnisse
# 1mi , time all  len of total labels in init vector 80, time_total 286.71242094039917 s
# 2 mi, len of total labels in init vector 79,  time_total 579.318208694458 = 9 min
# 5 mi, len of total labels in init vector 82 # time_total 2588.9190237522125 = 43 min
# 7 mi, len of total labels in init vector 79 # time_total 3572.8744649887085  = 1 hour

# last round with optics
# 2 milion data set in
##### Kmeans on last round
# 4mi : 68 index ; len(data_records) (40000, 100) , runtime total is 2915 s = 48 min
# only 40.000 element after processing in time_last 19.903489112854004 s
    # total time processsing is time_total 3503.8546810150146 = 58 min