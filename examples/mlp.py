"""Demonstrates MLP running on MNIST."""
import numpy as np
import pandas as pd
import pyopencl as cl
from pyopencl import array as clarray
from pyopencl import clrandom

import time
import sys
sys.path.append('..')

from shaderrider import clplatf as pl
from shaderrider import expr
from shaderrider import operators as op
from shaderrider import linalg
from get_data import get_mnist_data


class FullyConnectedLayer(object):
    def __init__(self, rng, li, input, nin, nout, W=None, b=None, activation_fn=op.Sigmoid):
        self.input = input
        q = pl.qs[0]
        if W is None:
            nw = np.asarray(rng.uniform(low=-np.sqrt(6./(nin+nout)),
                                        high=np.sqrt(6./(nin+nout)),
                                        size=(nin, nout)),
                            dtype=np.float32)
            if activation_fn == op.Sigmoid:
                nw *= 4
            W = clarray.to_device(q, nw)  #clrandom.rand(q, (nin, nout), np.float32, a=-1, b=1)
        if b is None:
            b = clarray.zeros(q, (nout,), np.float32)

        self.W = W
        self.b = b

        vW = expr.Variable('W'+li)
        vb = expr.Variable('b'+li)
        lin_out = op.Add(op.Dot(self.input, vW), vb)
        self.output = lin_out if activation_fn is None else activation_fn(lin_out)
        self.params = [(vW.name, self.W), (vb.name, self.b)]


class MLP(object):
    def __init__(self, rng, nin, nhid, nout):
        X = expr.Variable('X')
        Y = expr.Variable('Y')
        self.layer1 = FullyConnectedLayer(rng, '1', X, nin, nhid)
        self.layer2 = FullyConnectedLayer(rng, '2', self.layer1.output, nhid, nout)
        # self.layer2 = FullyConnectedLayer(rng, '2', X, nin, nout, activation_fn=None)

        self.y_pred = op.Argmax(self.layer2.output, -1)  # TODO test this axis
        self.cost = op.MeanSquaredErr(self.layer2.output, Y)  # op.Sub(self.layer2.output, Y)
        self.errors = op.Mean(op.NotEq(self.y_pred, Y))     # FIXME u nekim iteracijama greska je bila negativna!!
        self.params = self.layer1.params + self.layer2.params        # self.params = self.layer2.params

    def train(self, X, Y, learning_rate=0.01):
        val = pl.valuation()
        val['X'] = X
        val['Y'] = Y
        for name, value in self.params:
            val[name] = value

        grad = self.cost.rev_grad(val)

        debatch_help_vector = clarray.zeros(pl.qs[0], (Y.shape[0], 1),
                                            dtype=np.float32) + 1
        for name, value in self.params:
            if name.startswith('b'):
                dbh = linalg.dot(pl.qs[0], grad[name],
                                 debatch_help_vector, transA=True)
                value -= learning_rate*dbh.ravel()
            else:
                value -= learning_rate*grad[name]

    def test(self, X, Y):
        val = pl.valuation()
        val['X'] = X
        val['Y'] = Y
        for param, value in self.params:
            val[param] = value
        err = self.errors.evaluate(val)
        return err


def main():
    pl.init_cl(1)
    rng = np.random.RandomState(1234)
    mlp = MLP(rng, 784, 32, 10)

    tvX, tvY, testX, testY = get_mnist_data()
    tvX.shape = (60000, 784)
    testX.shape = (10000, 784)
    tvX = (tvX.astype(np.float32)/255.0).astype(np.float32)
    testX = (testX/255.0).astype(np.float32)
    tvYoh = pd.get_dummies(tvY).values.astype(np.float32)
    testYoh = pd.get_dummies(testY).values.astype(np.float32)
    vsplit = int(0.9 * tvX.shape[0])
    trainX = tvX[:vsplit, :]
    validX = tvX[vsplit:, :]
    trainY = tvYoh[:vsplit, :]
    validY = tvY[vsplit:].astype(np.float32)

    n_epochs = 30  # 000
    batch_size = 512
    n_train_batches = trainY.shape[0] / batch_size

    epoch = 0
    while epoch < n_epochs:
        epoch += 1
        for minibatch_index in xrange(n_train_batches):
            mlp.train(trainX[minibatch_index*batch_size:(minibatch_index+1)*batch_size, :],
                      trainY[minibatch_index*batch_size:(minibatch_index+1)*batch_size, :])

            if (minibatch_index % 5 == 0) and minibatch_index>0:
                err = mlp.test(validX, validY)
                print '\n>>>>>>>>>>>>>>>>>>>>>>>>>validation error:', err, 'batch:', minibatch_index, '/', n_train_batches

    for param, value in mlp.params:
        val = value.get()
        print '>PARAM', param, val.shape
        print val
        print '-'*79

if __name__ == '__main__':
    main()
