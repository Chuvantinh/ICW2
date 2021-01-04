# https://github.com/facebookresearch/faiss/wiki/Vector-codecs
import numpy as np
import faiss
# generate some data
x = np.random.rand(10000, 20).astype('float32')
print('x shape',x.shape)
print('x',x)
# prepare codec
codec = faiss.index_factory(20, "SQ4")

codec.train(x)
# encode
code = codec.sa_encode(x)
print(x.nbytes)
print('code',code)
print('code shape',code.shape)
# decode
x_decoded = codec.sa_decode(code[0:100])
print('x_decoded.shape', x_decoded.shape)
print('x_decoded', x_decoded)