"""Alexnet on cifar-10 example."""
import numpy as np
import pyopencl as cl
from pyopencl import array as clarray

import sys
sys.path.append('..')
from shaderrider import clplatf as pl
from shaderrider import expr
from shaderrider import operators as op


class ConvLayer(object):
    def __init__(self, label, rng, img, img_shape, filter_shape,
                 fstrides=(0, 0), zero_pad=(0, 0),
                 poolsize=(0, 0), activation_fn=None):
        q = pl.qs[0]
        self.img = img
        # TODO init W and b
        pf, ps = poolsize

        vW = expr.Variable('W'+label)
        vb = expr.Variable('b'+label)
        _out = op.Conv2d(self.img, vW, vb, strides=fstrides, zero_padding=zero_pad)
        if pf > 0 and ps > 0:
            _out = op.MaxPool(_out, pf, ps)
        self.output = activation_fn(_out)
        self.params = [self.W, self.b]


class FullyConnectedLayer(object):
    def __init__(self, rng, li, input, nin, nout, W=None, b=None,
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

        vW = expr.Variable('W'+li)
        vb = expr.Variable('b'+li)
        lin_out = op.Add(op.Dot(self.input, vW), vb)
        self.output = lin_out if activation_fn is None else activation_fn(lin_out)
        self.params = [(vW.name, self.W), (vb.name, self.b)]


class SoftmaxLayer(object):
    def __init__(self, label, rng, input, n_in, n_out, activation_fn=None):
        q = pl.qs[0]
        self.input = input
        self.W = clarray.zeros(q, (n_in, n_out), dtype=np.float32)
        self.b = clarray.zeros(q, (n_out,), dtype=np.float32)
        vW = expr.Variable('W'+label)
        vb = expr.Variable('b'+label)

        self.p_y_given_x = op.Softmax(op.Add(op.Dot(self.input, vW), vb))
        self.y_pred = op.Argmax(self.p_y_given_x, axis=-1)

        self.params = [self.W, self.b]


class Alexnet(object):
    def __init__(self, rng, img_shape, n_out):
        X = expr.Variable('X')
        Y = expr.Variable('Y')
        self.layer1 = ConvLayer('_C1', rng, X, img_shape, ())


def main():
    pass


if __name__ == '__main__':
    main()
