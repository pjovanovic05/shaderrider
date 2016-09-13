"""Alexnet on cifar-10 example."""
import numpy as np
import pandas as pd
import pyopencl as cl
from pyopencl import array as clarray

import sys
sys.path.append('..')
from shaderrider import clplatf as pl
from shaderrider import expr
from shaderrider import operators as op
from shaderrider.utils import misc


class ConvLayer(object):
    def __init__(self, label, rng, img, img_shape, filter_shape,
                 fstrides=(0, 0), zero_pad=(0, 0),
                 poolsize=(0, 0), activation_fn=None):
        q = pl.qs[0]
        self.img = img
        # TODO init W and b
        fan_in = np.prod(filter_shape[1:])
        fan_out = filter_shape[0]*np.prod(filter_shape[2:])
        W_bound = np.sqrt(6./(fan_in+fan_out))
        nW = np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=np.float32)
        print '>>>start CW1:\n', nW
        self.W = clarray.to_device(q, nW)
        self.b = clarray.zeros(q, (filter_shape[0],), dtype=np.float32)

        pf, ps = poolsize

        vW = expr.Variable('W'+label)
        vb = expr.Variable('b'+label)
        _out = op.Conv2d(self.img, vW, vb, strides=fstrides, zero_padding=zero_pad)
        if pf > 0 and ps > 0:
            _out = op.MaxPool(_out, pf, ps)
        self.output = activation_fn(_out)
        self.params = [(vW.name, self.W), (vb.name, self.b)]


class FullyConnectedLayer(object):
    def __init__(self, label, rng, input, nin, nout, W=None, b=None,
                 activation_fn=op.Sigmoid):
        q = pl.qs[0]
        self.input = input
        if W is None:
            nw = np.asarray(rng.uniform(low=-np.sqrt(6./(nin+nout)),
                                        high=np.sqrt(6./(nin+nout)),
                                        size=(nin, nout)),
                            dtype=np.float32)
            if activation_fn == op.Sigmoid:
                nw *= 4
            W = clarray.to_device(q, nw)
        if b is None:
            b = clarray.zeros(q, (nout,), np.float32)

        self.W = W
        self.b = b

        vW = expr.Variable('W'+label)
        vb = expr.Variable('b'+label)
        lin_out = op.Add(op.Dot(self.input, vW), vb)
        self.output = lin_out if activation_fn is None else activation_fn(lin_out)
        self.params = [(vW.name, self.W), (vb.name, self.b)]


class SoftmaxLayer(object):
    def __init__(self, label, rng, input, n_in, n_out):
        q = pl.qs[0]
        self.input = input
        self.W = clarray.zeros(q, (n_in, n_out), dtype=np.float32)
        self.b = clarray.zeros(q, (n_out,), dtype=np.float32)
        vW = expr.Variable('W'+label)
        vb = expr.Variable('b'+label)

        self.p_y_given_x = op.Softmax(op.Add(op.Dot(self.input, vW), vb))
        self.y_pred = op.Argmax(self.p_y_given_x, axis=-1)

        self.params = [(vW.name, self.W), (vb.name, self.b)]


class Alexnet(object):
    def __init__(self, rng, img_shape, n_out):
        X = expr.Variable('X')
        Y = expr.Variable('Y')
        self.layer1 = ConvLayer('_C1', rng, X, img_shape, (64, 3, 3, 3),
                                fstrides=(1, 1), zero_pad=(1, 1), poolsize=(2,2),
                                activation_fn=op.ReLU)
        nin = 64*16*16
        reshaped_conv_out = op.Reshape(self.layer1.output, (-1, nin))
        self.layer2 = FullyConnectedLayer('F1', rng, reshaped_conv_out, nin, 100)
        self.layer3 = SoftmaxLayer('S1', rng, self.layer2.output, 100, 10)
        self.cost = op.MeanSquaredErr(self.layer3.p_y_given_x, Y)
        self.error = op.Mean(op.NotEq(self.layer3.y_pred, Y))
        self.params = self.layer1.params + self.layer2.params + self.layer3.params

    def train(self, X, Y, learning_rate=0.01):
        val = pl.valuation()
        val['X'] = X
        val['Y'] = Y
        for name, value in self.params:
            val[name] = value

        grad = self.cost.rev_grad(val)
        # print grad
        for name, value in self.params:
            # print 'updating', name
            # print 'shape:', value.shape, 'grad shape:', grad[name].shape
            if name.startswith('bF') or name.startswith('bS'):
                bgsum = misc.sum(pl.qs[0], grad[name], axis=0)
                value -= learning_rate*bgsum
            else:
                value -= learning_rate*grad[name]

    def test(self, X, Y):
        val = pl.valuation()
        val['X'] = X
        val['Y'] = Y
        for param, value in self.params:
            val[param] = value
        err = self.error.evaluate(val)
        return err


def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


def main():
    pl.init_cl(1)
    rng = np.random.RandomState(1234)
    anet = Alexnet(rng, (10, 3, 32, 32), 10)

    db1 = unpickle('/home/petar/datasets/cifar-10-batches-py/data_batch_1')
    tdb = unpickle('/home/petar/datasets/cifar-10-batches-py/test_batch')
    X = db1['data'].reshape(10000, 3, 32, 32).astype(np.float32)/255.0
    Y = db1['labels']
    trainY = pd.get_dummies(Y).values.astype(np.float32)
    tX = tdb['data'].reshape(-1, 3, 32, 32).astype(np.float32)/255.0
    tY = np.asarray(tdb['labels'], dtype=np.float32)

    n_epochs = 1
    batch_size = 128
    n_train_batches = trainY.shape[0] / batch_size

    epoch = 0
    while epoch < n_epochs:
        epoch += 1
        g = None
        for mbi in xrange(n_train_batches):
            anet.train(X[mbi*batch_size:(mbi+1)*batch_size, :], trainY[mbi*batch_size:(mbi+1)*batch_size, :])
        print '='*70
        print '>>final wc:\n', anet.layer1.params[0][1]
    print 'test error:'
    print anet.test(tX, tY)


if __name__ == '__main__':
    main()
