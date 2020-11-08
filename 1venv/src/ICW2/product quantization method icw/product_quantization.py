import pqkmeans
import numpy as np
X = np.random.random((1000, 128)) # 128 dimensional 100,000 samples
print(X)
# Train a PQ encoder.
# Each vector is divided into 4 parts and each part is
# encoded with log256 = 8 bit, resulting in a 32 bit PQ code.
encoder = pqkmeans.encoder.PQEncoder(num_subdim=4, Ks=256)
encoder.fit(X[:1000])  # Use a subset of X for training

# Convert input vectors to 32-bit PQ codes, where each PQ code consists of four uint8.
# You can train the encoder and transform the input vectors to PQ codes preliminary.
X_pqcode = encoder.transform(X)

# Run clustering with k=5 clusters.
kmeans = pqkmeans.clustering.PQKMeans(encoder=encoder, k=5)
clustered = kmeans.fit_predict(X_pqcode)

# Then, clustered[0] is the id of assigned center for the first input PQ code (X_pqcode[0]).

# Try with kmeans
#clustered_kmeans = KMeans(n_clusters=K, n_jobs=-1).fit_predict(X4)
