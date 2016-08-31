"""Demonstrates MLP running on MNIST."""
import numpy as np
import pyopencl as cl
from pyopencl import array as clarray
from pyopencl import clrandom

from shaderrider.clplatf import valuation, init_cl
from shaderrider import expr
from shaderrider import operators as op


class FullyConnectedLayer(object):
    def __init__(self, input, nin, nout, W=None, b=None, activation_fn=op.sigmoid):
        self.input = input

        if W is None:
            W = expr.Variable('W')   # TODO

        if b is None:
            b = expr.Variable('b')

        self.W = W
        self.b = b

        vW = expr.Variable('W')
        vb = expr.Variable('b')
        lin_out = op.Add(op.Dot(input, vW), vb)
        self.output = lin_out if activation_fn is None else activation_fn(lin_out)


class MLP(object):
    def __init__(self, input, nin, nhid, nout):
        self.layer1 = FullyConnectedLayer(input, nin, nhid)
        self.layer2 = FullyConnectedLayer(self.layer1.output, nhid, nout)
        self.cost = op.MeanSquareErr(self.layer2.output, Y)     # TODO oklen Y?

    def train(self, X, Y):
        # output poslednjeg layera je outf. treba mi error function za to...
        pass

    def classify(self):
        pass


def main():
    pass


if __name__ == '__main__':
    main()
