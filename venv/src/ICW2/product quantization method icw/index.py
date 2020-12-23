#https://towardsdatascience.com/comprehensive-guide-to-approximate-nearest-neighbors-algorithms-8b94f057d6b6
import pickle
import faiss

class ExactIndex():
    def __init__(self, vectors, labels):
        self.dimension = vectors.shape[1]
        self.vectors = vectors.astype('float32')
        self.labels = labels

    def build(self):
        self.index = faiss.IndexFlatL2(self.dimension, )
        self.index.add(self.vectors)

    def query(self, vectors, k=10):
        print(vectors.shape)
        distances, indices = self.index.search(vectors, k)

        print(distances)
        # I expect only query on one vector thus the slice
        #return [self.labels[i] for i in indices[0]]

from sklearn.datasets.samples_generator import make_blobs
import numpy as np
#centers = [[2, 2], [8, 9], [9, 5], [3,9]]
#X, y =make_blobs(n_samples=1000, n_features=2, centers=centers, cluster_std=0.5, center_box=(1, 10.0), shuffle=True, random_state=0)

n_points_per_cluster_total = 1000
size_colum = 100
centers = np.random.randint(-20, 20, size=(size_colum,size_colum))
# train
X, y = make_blobs(n_samples=n_points_per_cluster_total,
                    centers=centers,
                    n_features=size_colum,
                    cluster_std=0.4,
                    random_state=0)

print(X[0].shape)
#print(y)
import matplotlib.pyplot as plt
# Plot the training points
plt.scatter(X[:, 0], X[:, 1])
plt.xlabel('X axis')
plt.ylabel('X axis')
plt.show()
array_name = np.arange(100)
data = {
    "name": array_name,
    "vector": X
}

index = ExactIndex(data["vector"], data["name"])
index.build()
index.query(data['vector'][0])