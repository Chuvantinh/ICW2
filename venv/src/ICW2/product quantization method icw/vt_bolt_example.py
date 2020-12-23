# https://github.com/dblalock/bolt
import bolt
import numpy as np
from scipy.stats import pearsonr as corr
from sklearn.datasets import load_digits
import timeit

# for simplicity, use the sklearn digits dataset; we'll split
# it into a matrix X and a set of queries Q
X, _ = load_digits(return_X_y=True)
nqueries = 20
X, Q = X[:-nqueries], X[-nqueries:]

enc = bolt.Encoder(reduction='dot', accuracy='lowest') # can tweak acc vs speed
enc.fit(X)

dot_corrs = np.empty(nqueries)
for i, q in enumerate(Q):
    dots_true = np.dot(X, q)
    dots_bolt = enc.transform(q)
    dot_corrs[i] = corr(dots_true, dots_bolt)[0]

# dot products closely preserved despite compression
print ("dot product correlation: {} +/- {}".format(
    np.mean(dot_corrs), np.std(dot_corrs)))  # > .97

# massive space savings
print(X.nbytes)  # 1777 rows * 64 cols * 8B = 909KB
print(enc.nbytes)  # 1777 * 2B = 3.55KB

# massive time savings (~10x here, but often >100x on larger
# datasets with less Python overhead; see the paper)
t_np = timeit.Timer(
    lambda: [np.dot(X, q) for q in Q]).timeit(5)        # ~9ms
t_bolt = timeit.Timer(
    lambda: [enc.transform(q) for q in Q]).timeit(5)    # ~800us
print ("Numpy / BLAS time, Bolt time: {:.3f}ms, {:.3f}ms".format(
    t_np * 1000, t_bolt * 1000))

# can get output without offset/scaling if needed
dots_bolt = [enc.transform(q, unquantize=True) for q in Q]