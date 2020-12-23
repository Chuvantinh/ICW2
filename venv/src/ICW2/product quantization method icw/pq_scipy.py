from scipy.cluster.vq import kmeans2, vq
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import time
from sklearn.cluster import KMeans

def compute_code_books(vectors, sub_size=2, n_cluster=128, n_iter=20, minit='points', seed=123):
    n_rows, n_cols = vectors.shape
    n_sub_cols = n_cols // sub_size

    np.random.seed(seed)
    code_books = np.zeros((sub_size, n_cluster, n_sub_cols), dtype=np.float32)
    for subspace in range(sub_size):
        sub_vectors = vectors[:, subspace * n_sub_cols:(subspace + 1) * n_sub_cols]
        centroid, label = kmeans2(sub_vectors, n_cluster, n_iter, minit=minit)
        code_books[subspace] = centroid

    return code_books

sub_size = 5  # m
n_cluster = 100  # k

#dataset
n_points_per_cluster_total = 500000
size_colum = 100
centers = np.random.randint(-20, 20, size=(size_colum,size_colum))

X1, labels_true = make_blobs(n_samples=n_points_per_cluster_total,
                            centers=centers,
                            n_features=size_colum,
                            cluster_std=0.4,
                            random_state=0)

# learning the cluster centroids / code books for our output matrix/embedding
code_books = compute_code_books(X1, sub_size, n_cluster)
print('code book size: ', code_books.shape)
#code book size:  (2, 64, 40)

def encode(vectors, code_books):
    n_rows, n_cols = vectors.shape
    sub_size = code_books.shape[0]
    n_sub_cols = n_cols // sub_size # 20

    codes = np.zeros((n_rows, sub_size), dtype=np.int32)
    sub_vectors_get = []
    for subspace in range(sub_size):
        # subspace: 0,1 2 3 4
        sub_vectors = vectors[:, subspace * n_sub_cols:(subspace + 1) * n_sub_cols]
        code, dist = vq(sub_vectors, code_books[subspace])
        codes[:, subspace] = code
        sub_vectors_get.append(sub_vectors)
    return codes, sub_vectors_get
# our original embedding now becomes the cluster centroid for each subspace
print("Data type of each element X1 :\n{}\n".format(type(X1[0][0])))
vector_codes, sub_vectors = encode(X1, code_books)
print('encoded vector codes size: ', vector_codes.shape)
print('vector_codes element: ', vector_codes)
print("Data type of each element sub_vectors :\n{}\n".format(type(sub_vectors[1][1])))
#
# print('sub_vectors: ', sub_vectors[:1])
print('sub_vectors LEN: ', len(sub_vectors))
print ('X1 : ',X1)
print('sub_vectors 0 shap: ',sub_vectors[0].shape)
print('sub_vectors 0: ',sub_vectors[0].shape)
print('sub_vectors 1: ',sub_vectors[1].shape)
print('sub_vectors 2: ',sub_vectors[2])
print('sub_vectors 3: ',sub_vectors[3])
print('sub_vectors 4: ',sub_vectors[4])
############# KMEANS #################
db_time = time.time()
kmeans = KMeans(n_clusters=100, random_state=0, n_jobs=-1).fit(sub_vectors[0])
kmeans.predict(sub_vectors[0])

db_time_process = time.time() - db_time
print('Number of cluster in  Kmeans :  % s ' % len(kmeans.cluster_centers_))
print('Elapsed time to cluster in kmeans :  %.4f s ' % db_time_process)
############# ENDS KMEANS #################

############# PQ KMEANS #################
db_time_kmeans = time.time()
kmeans = KMeans(n_clusters=100, random_state=0, n_jobs=-1).fit(sub_vectors[1])
kmeans.predict(sub_vectors[1])

db_time_process = time.time() - db_time_kmeans
print('Number of cluster in  Kmeans :  % s ' % len(kmeans.cluster_centers_))
print('Elapsed time to cluster in kmeans :  %.4f s ' % db_time_process)
############# ENDS KMEANS #################

############# DBSCANN #################
# Run data encoder with dbscan ball tree is the best
from sklearn.cluster import DBSCAN
db_time_dbscan = time.time()
db = DBSCAN(eps=10,algorithm='ball_tree',leaf_size=10, min_samples=100, n_jobs = -1).fit(sub_vectors[2])
db_time_dbscan_process = time.time() - db_time_dbscan
#array false for core samples mask
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
#set for db.core sample indices as true
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Number of cluster in DBSCANN :  % d ' % n_clusters_)
print('Elapsed time to cluster in DBSCANN hihi :  %.4f s ' % db_time_dbscan_process)
############# END DBSCANN #################

############# Begin Optics #################
# from sklearn.cluster import OPTICS, cluster_optics_dbscan
# op_time = time.time()
# clust = OPTICS(min_samples=50, xi=.05, min_cluster_size=.05, algorithm="kd_tree", n_jobs=-1).fit(sub_vectors[3])
# reachability = clust.reachability_[clust.ordering_]
# labels = clust.labels_[clust.ordering_]
# op_time_processing = time.time() - op_time
# op_labels = clust.labels_
# n_clusters_op_ = len(set(op_labels)) - (1 if -1 in op_labels else 0)
# print('Cluster sind ', n_clusters_op_)
# print('Elapsed time to cluster in Optics :  %.4f s ' % op_time_processing)
# ############# END Optics #################

############# Begin PKMEANS #################
############# END PKMEANS #################


#print('sub_vectors shape: ', sub_vectors.shape)
#vector_codes
# encoded vector codes size:  (733, 2)
# array([[34, 61],
#        [29, 19],
#        [ 7,  4],
#        ...,
#        [42, 52],
#        [42, 47],
#        [42, 52]], dtype=int32)