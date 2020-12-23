#https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/
from sklearn.preprocessing import StandardScaler
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
n_points_per_cluster_total = 1000
size_colum = 100
centers = np.random.randint(-20, 20, size=(size_colum,size_colum))
#print('centers', centers[0])
# train
X1, labels_true = make_blobs(n_samples=n_points_per_cluster_total,
                            centers=centers,
                            n_features=size_colum,
                            cluster_std=0.4,
                            random_state=0)
print(X1)
print(X1[0])
print(X1[999].shape)
# dataset do it
X2, labels_2 = make_blobs(n_samples=1000,
                            centers=centers,
                            n_features=size_colum,
                            cluster_std=0.4,
                            random_state=0)

scaler = StandardScaler()
scaler.fit(X1)

X_train = scaler.transform(X1)
X_test = scaler.transform(X2)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, labels_true)

y_pred = classifier.predict(X_test)
print(y_pred)
from sklearn.metrics import classification_report, confusion_matrix
#print(confusion_matrix(labels_2, y_pred))
#print(classification_report(labels_2, y_pred))