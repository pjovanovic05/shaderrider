import unittest
import numpy as np
from pyopencl import array as clarray

from shaderrider import clplatf
from shaderrider import linalg


class LinalgTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        clplatf.init_cl(1)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_dot(self):
        q = clplatf.qs[0]
        X = np.random.uniform(0, 1, (10,)).astype(np.float32)
        Y = np.random.uniform(0, 1, (10,)).astype(np.float32)
        expected = np.dot(X, Y)
        gX = clarray.to_device(q, X)
        gY = clarray.to_device(q, Y)

        gR, ev = linalg.dot(q, gX, gY)
        ev.wait()
        R = gR.get()
        self.assertTrue(np.allclose(R, expected))
