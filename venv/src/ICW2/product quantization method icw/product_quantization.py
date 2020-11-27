import pqkmeans
import numpy as np
from sklearn.cluster import KMeans
# for data set
from sklearn.datasets.samples_generator import make_blobs
# time measure
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

n_points_per_cluster_total = 1000
size_colum = 100
centers = np.random.randint(-20, 20, size=(size_colum,size_colum))

# train
X1, labels_true = make_blobs(n_samples=n_points_per_cluster_total,
                            centers=centers,
                            n_features=size_colum,
                            cluster_std=0.4,
                            random_state=0)
# dataset do it
X2, labels_2 = make_blobs(n_samples=1000000,
                            centers=centers,
                            n_features=size_colum,
                            cluster_std=0.4,
                            random_state=0)

# Train a PQ encoder.
# Each vector is divided into 4 parts and each part is
# encoded with log256 = 8 bit, resulting in a 32 bit PQ code.
encoder = pqkmeans.encoder.PQEncoder(num_subdim=5, Ks=256)
# encoder train is (5 subpspaces * 256 codewords * 20 dimensions):
encoder.fit(X1)  # Use a subset of X for training
# using for large dataset : fit_generator
print("codewords.shape:\n{}".format(encoder.codewords.shape))
# Convert input vectors to 32-bit PQ codes, where each PQ code consists of four uint8.
# You can train the encoder and transform the input vectors to PQ codes preliminary.

# transform big to small dimension
X_pqcode = encoder.transform(X2)
#X_reconstructed = encoder.inverse_transform(X_pqcode)
#print("X_reconstructed.shap: ")
#print(X_reconstructed.shape)
print("X_pqcode.shape")
print(X_pqcode.shape)
#print(X_pqcode)
print('convert codeword to original: ')



# Run clustering with k=5 clusters.
# db_time_pqkmean = time.time()
# pqkmeans_cluster = pqkmeans.clustering.PQKMeans(encoder=encoder, k=100)
# clustered = pqkmeans_cluster.fit_predict(X_pqcode)
# db_time_pqkmean_process = time.time() - db_time_pqkmean
# print('Number of cluster in PQ Kmeans :  % s ' % len(pqkmeans_cluster.cluster_centers_))
# print('Elapsed time to cluster in PQ Kmeans :  %.4f s ' % db_time_pqkmean_process)

X_reconstructed = encoder.inverse_transform(X_pqcode)
#print("X: \n")
#print(X)
print("X_reconstructed: \n")
print(X_reconstructed.shape)
#print(X_reconstructed)
print("------------------------------")
print("Encode of pqkmeans_cluster.cluster_centers_ :")
##################### DIVIDE X_reconstructed (20000, 100) #### and run with db scan optics hdbscan
# because after reconstructure the data stored with dtype=float64. it means ram can store more dataset and helpful for algorithment

# Run data encoder with dbscan
#ball tree is the best
from sklearn.cluster import DBSCAN
db_time_dbscan = time.time()
db = DBSCAN(eps=10,algorithm='ball_tree',leaf_size=10, min_samples=100, n_jobs = -1).fit(X_reconstructed)
db_time_dbscan_process = time.time() - db_time_dbscan
#array false for core samples mask
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
#set for db.core sample indices as true
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Number of cluster in DBSCANN :  % d ' % n_clusters_)
print('Elapsed time to cluster in DBSCANN :  %.4f s ' % db_time_dbscan_process)

#print(pqkmeans_cluster.cluster_centers_)
#print("X PQ centers reconstructed: \n")
#clustering_centers_numpy = np.array(pqkmeans_cluster.cluster_centers_, dtype=encoder.code_dtype)  # Convert to np.array with the proper dtype
#clustering_centers_reconstructd = encoder.inverse_transform(clustering_centers_numpy) # From PQ-code to 6D vectors
#print(clustering_centers_reconstructd)

# db_time = time.time()
# kmeans = KMeans(n_clusters=100, random_state=0, n_jobs=-1).fit(X_pqcode)
# kmeans.predict(X_pqcode)
#
# db_time_process = time.time() - db_time
# print('Number of cluster in  Kmeans :  % s ' % len(kmeans.cluster_centers_))
# print('Elapsed time to cluster in kmeans :  %.4f s ' % db_time_process)

# Then, clustered[0] is the id of assigned center for the first input PQ code (X_pqcode[0]).

# Try with kmeans
#clustered_kmeans = KMeans(n_clusters=K, n_jobs=-1).fit_predict(X4)
