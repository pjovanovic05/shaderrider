"""Test neural net utility functions module."""
import unittest
import numpy as np
import pyopencl as cl
from pyopencl import array as clarray

from shaderrider import clplatf as pl
from shaderrider import nnet
# import sys


class NnetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pl.init_cl(1)

    def test_argmax(self):
        n = 20
        Y = np.zeros((n, 10), dtype=np.float32)
        expected = np.empty((n,), dtype=np.float32)
        ri = 0
        for i in range(Y.shape[0]):
            Y[i, ri] = 1
            expected[i] = ri
            ri = (ri+1) % 10

        gY = clarray.to_device(pl.qs[0], Y)
        # print >>sys.stderr, 'gYshape:', gY.shape, "strides:", gY.strides
        gamax, ev = nnet.argmax(pl.qs[0], gY, 1)
        ev.wait()

        amax = gamax.get()
        # print >>sys.stderr, '\namax:\n',amax, amax.shape
        # print >>sys.stderr, 'expected:\n', expected
        self.assertTrue(np.allclose(amax, expected))
