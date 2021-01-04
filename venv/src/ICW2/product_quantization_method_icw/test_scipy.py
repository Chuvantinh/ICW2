# import numpy as np
# from scipy.cluster.vq import vq, kmeans2
# from scipy.spatial.distance import cdist
#
#
# def train(vec, M):
#     Ds = int(vec.shape[1] / M)  # Ds = D / M
#     # codeword[m][k] = cmk
#     codeword = np.empty((M, 256, Ds), np.float32)
#     for m in range(M):
#         vec_sub = vec[:, m * Ds: (m + 1) * Ds]
#     codeword[m], label = kmeans2(vec_sub, 256)
#     return codeword
#
# def encode(codeword, vec):
#     M, _K, Ds = codeword.shape
#     pqcode = np.empty((vec.shape[0], M), np.uint8)
#     for m in range(M):  # Eq. (3) and Eq. (4)
#         vec_sub = vec[:, m * Ds: (m + 1) * Ds]
#         pqcode[:m], dist = vq(vec_sub, codeword[m]
#     return pqcode
#
# def search(codeword, pqcode, query):
#     M, _K, Ds = codeword.shape
#     # dist_table = A(m,k)
#     dist_table = np.empty((M, 256), np.float32)
#     for m in range(M):
#         query_sub = query[m * Ds: (m + 1) * Ds]
#     dist_table[m, :] = cdist([query_sub],
#                              codeword[m], 'sqeuclidean')[0]
#     dist = np.sum(dist_table[range(M), pqcode], axis=1)
#     return dist
#
# if __name__ == "__main__":
# # Read vec_train, vec ({xn}Nn=1), and query (y)
#     M = 4
#     codeword = train(vec_train, M)
#     pqcode = encode(codeword, vec)
#     dist = search(codeword, pqcode, query)
#     print(dist)
