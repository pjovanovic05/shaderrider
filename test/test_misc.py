import unittest
import numpy as np
import pyopencl as cl
from pyopencl import array as clarray

from shaderrider import clplatf
from shaderrider.utils import misc


class MiscTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        clplatf.init_cl(1)

    def test_max(self):
        ax = 1
        q = clplatf.qs[0]
        X = np.arange(1280).astype(np.float32)
        X.shape = (128, 10)
        gX = clarray.to_device(q, X)
        gmax, gidx = misc.max(q, gX, axis=ax, keepdims=True)
        am = gmax.get()
        em = X.max(axis=ax, keepdims=True)
        self.assertTrue(np.allclose(am, em))
