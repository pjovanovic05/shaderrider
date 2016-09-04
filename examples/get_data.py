"""Fetches and loads the datasets."""
import urllib
import os
import gzip
import struct
import numpy as np
from array import array


def get_mnist_data():
    mnist_base = 'http://yann.lecun.com/exdb/mnist/'
    mnist_files = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
                   't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
    # get the data
    for fname in mnist_files:
        if not os.path.isfile(fname):
            print 'fetching', fname
            urllib.urlretrieve(mnist_base + fname, fname)
    # load data
    outs = []
    for i, tf in enumerate(mnist_files):
        print 'loading', tf
        with gzip.open(tf, 'rb') as f:
            magic, n = struct.unpack('>II', f.read(8))
            if magic == 2049:
                # labels...
                lbls = array('b', f.read())
                arry = np.frombuffer(lbls, dtype=np.uint8)
                arry.shape = (n,)
                outs.append(arry)
            elif magic == 2051:
                # images
                w, h = struct.unpack('>II', f.read(8))
                imgs = array('b', f.read())
                arry = np.frombuffer(imgs, dtype=np.uint8)
                arry.shape = (n, h, w)
                outs.append(arry)
    return outs
