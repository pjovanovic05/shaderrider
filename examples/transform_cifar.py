#!/usr/bin/env python
import cPickle
import numpy as np
import pandas as pd


def load_cifar(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


def save_dataset(trainX, trainY, testX, testY, filename):
    with open(filename, 'wb') as f:
        cPickle.dump([(trainX, trainY), (testX, testY)], f)


def main():
    db1 = load_cifar('/home/petar/datasets/cifar-10-batches-py/data_batch_1')
    db2 = load_cifar('/home/petar/datasets/cifar-10-batches-py/data_batch_2')
    db3 = load_cifar('/home/petar/datasets/cifar-10-batches-py/data_batch_3')
    db4 = load_cifar('/home/petar/datasets/cifar-10-batches-py/data_batch_4')
    db5 = load_cifar('/home/petar/datasets/cifar-10-batches-py/data_batch_5')
    tdb = load_cifar('/home/petar/datasets/cifar-10-batches-py/test_batch')
    X1 = db1['data'].reshape(10000, 3, 32, 32).astype(np.float32)/255.0
    Y1 = db1['labels']
    trainY1 = pd.get_dummies(Y1).values.astype(np.float32)
    X2 = db2['data'].reshape(10000, 3, 32, 32).astype(np.float32)/255.0
    Y2 = db2['labels']
    trainY2 = pd.get_dummies(Y2).values.astype(np.float32)
    X3 = db3['data'].reshape(10000, 3, 32, 32).astype(np.float32)/255.0
    Y3 = db3['labels']
    trainY3 = pd.get_dummies(Y3).values.astype(np.float32)
    X4 = db4['data'].reshape(10000, 3, 32, 32).astype(np.float32)/255.0
    Y4 = db4['labels']
    trainY4 = pd.get_dummies(Y4).values.astype(np.float32)
    X5 = db5['data'].reshape(10000, 3, 32, 32).astype(np.float32)/255.0
    Y5 = db5['labels']
    trainY5 = pd.get_dummies(Y5).values.astype(np.float32)

    tX = tdb['data'].reshape(-1, 3, 32, 32).astype(np.float32)/255.0
    tY = np.asarray(tdb['labels'], dtype=np.float32)

    X = np.concatenate((X1, X2, X3, X4, X5), axis=0)
    Y = np.concatenate((Y1, Y2, Y3, Y4, Y5), axis=0)
    print 'X shape:', X.shape
    print 'Y shape:', Y.shape
    print 'test shape:', tX.shape, tY.shape
    save_dataset(X, Y, tX, tY, 'cifar-10.pkl')
    print 'DONE'


if __name__ == '__main__':
    main()
