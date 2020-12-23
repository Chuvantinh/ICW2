# https://towardsdatascience.com/using-faiss-to-search-in-multidimensional-spaces-ccc80fcbf949
# subvector o day no get kieu gi nhi ?
#Flat indexes just encode the vectors into codes of a fixed size and store them in an array of ntotal * code_size bytes.
# At search time, all the indexed vectors are decoded sequentially and compared to the query vectors. For the IndexPQ the comparison is done in the compressed domain, which is faster.

# tutorial : https://github.com/facebookresearch/faiss/wiki/Getting-started?fbclid=IwAR2s4zKxvZzbPZsfAjYUSYv1soJjZk4t59VROFnMR-ed0aLcBskt48J9vlA

# wiki: https://en.wikipedia.org/wiki/Vector_quantization
import numpy as np
d = 64                           # dimension
nb = 100000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.
import faiss

nlist = 100
m = 8
k = 4
quantizer = faiss.IndexFlatL2(d)  # this remains the same
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)# 8 specifies that each sub-vector is encoded as 8 bits
index.train(xb)
index.add(xb)
D, I = index.search(xb[:5], k) # sanity check
print('In dex',I)
print('Distances',D)
index.nprobe = 10           # make comparable with experiment above
D, I = index.search(xq, k)     # search
print('I[-5:]',I[-5:])
# n_points_per_cluster_total = 1000
# size_colum = 100
# centers = np.random.randint(-20, 20, size=(size_colum,size_colum))
# # train
# X, y = make_blobs(n_samples=n_points_per_cluster_total,
#                     centers=centers,
#                     n_features=size_colum,
#                     cluster_std=0.4,
#                     random_state=0)
#
# print(X[0].shape)
# index = faiss.IndexFlatL2(X.astype('float32'))   # build the index
# print(index.is_trained)
#index.add(xb)                  # add vectdors to the index
#print(index.ntotal)