import pqkmeans
import numpy as np
from sklearn.cluster import KMeans
# for data set
from sklearn.datasets.samples_generator import make_blobs
# time measure
import time
db_time_pqkmean = time.time()
n_points_per_cluster_total = 100000
size_colum = 100
centers = np.random.randint(-20, 20, size=(size_colum,size_colum))
#print('centers', centers[0])
# train to create code and codebook
X1, labels_true = make_blobs(n_samples=n_points_per_cluster_total,
                            centers=centers,
                            n_features=size_colum,
                            cluster_std=0.4,
                            random_state=0)
# dataset  to cluster.
X2, labels_2 = make_blobs(n_samples=8000000,
                            centers=centers,
                            n_features=size_colum,
                            cluster_std=0.4,
                            random_state=0)
print("X2 shape", X2.shape)
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
print("X_pqcode",X_pqcode)
# Run clustering with k=5 clusters.

pqkmeans_cluster = pqkmeans.clustering.PQKMeans(encoder=encoder, k=100)
clustered = pqkmeans_cluster.fit_predict(X_pqcode)
db_time_pqkmean_process = time.time() - db_time_pqkmean
print('Number of cluster in PQ Kmeans :  % s ' % len(pqkmeans_cluster.cluster_centers_))
print('Elapsed time to cluster in PQ Kmeans :  %.4f s ' % db_time_pqkmean_process)

#X_reconstructed1 = encoder.inverse_transform(X_pqcode[0:2000])
#X_reconstructed2 = encoder.inverse_transform(X_pqcode[2000:4000])
#print("X: \n")
#print(X)
#print("Shape ofX_reconstructed1: \n", X_reconstructed1.shape)
#print('X_reconstructed1', X_reconstructed1)
#print("------------------------------")
#print("Encode of pqkmeans_cluster.cluster_centers_ :")