"""Demonstrates MLP running on MNIST."""
import numpy as np
import pandas as pd
import pyopencl as cl
from pyopencl import array as clarray
from pyopencl import clrandom

import sys
sys.path.append('..')

from shaderrider import clplatf as pl
from shaderrider import expr
from shaderrider import operators as op
from get_data import get_mnist_data


class FullyConnectedLayer(object):
    def __init__(self, li, input, nin, nout, W=None, b=None, activation_fn=op.Sigmoid):
        self.input = input
        q = pl.qs[0]
        if W is None:
            W = clrandom.rand(q, (nin, nout), np.float32)
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
    def __init__(self, nin, nhid, nout):
        X = expr.Variable('X')
        Y = expr.Variable('Y')
        self.layer1 = FullyConnectedLayer('1', X, nin, nhid)
        self.layer2 = FullyConnectedLayer('2', self.layer1.output, nhid, nout)

        self.y_pred = op.Argmax(self.layer2.output, -1)  # TODO test this axis
        self.cost = op.MeanSquaredErr(self.layer2.output, Y)
        self.errors = op.Mean(op.NotEq(self.y_pred, Y))
        self.params = self.layer1.params + self.layer2.params

    def train(self, X, Y, learning_rate=0.01):
        val = pl.valuation()
        val['X'] = X
        val['Y'] = Y
        for name, value in self.params:
            val[name] = value

        grad = self.cost.rev_grad(val)
        for name, value in self.params:
            value -= learning_rate * grad[name]

    def test(self, X, Y):
        val = pl.valuation(784, 1024, 10)
        val['X'] = X
        val['Y'] = Y
        for param, value in self.params:
            val[param] = value
        err = self.errors(val)
        return err


def main():
    pl.init_cl(1)
    mlp = MLP(784, 1024, 10)

    tvX, tvY, testX, testY = get_mnist_data()
    tvX = (tvX/255.0).astype(np.float32)
    testX = (testX/255.0).astype(np.float32)
    tvYoh = pd.get_dummies(tvY).values
    testYoh = pd.get_dummies(testY).values
    vsplit = int(0.9 * tvX.shape[0])
    trainX = tvX[:vsplit, :]
    validX = tvX[vsplit:, :]
    trainY = tvYoh[:vsplit, :]
    validY = tvYoh[vsplit:, :]

    n_epochs = 1  # 000
    batch_size = 512
    n_train_batches = trainY.shape[0] / batch_size

    epoch = 0
    while epoch < n_epochs:
        epoch += 1
        for minibatch_index in xrange(n_train_batches):
            mlp.train(trainX[minibatch_index*batch_size:(minibatch_index+1)*batch_size, :],
                      trainY[minibatch_index*batch_size:(minibatch_index+1)*batch_size, :])
            print 'train batch:', minibatch_index

    # TODO convert training data into batches
    # iterate over batches calling train
    # on each n-th iteration, call validation
    # when the validation error improvement is low enough break out
    # give training error results


if __name__ == '__main__':
    main()
