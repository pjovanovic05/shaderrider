import unittest
import numpy as np
import pyopencl as cl
from pyopencl import array as clarray

from shaderrider import clplatf
from shaderrider import conv

import sys


class ConvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        clplatf.init_cl(1)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_im2col(self):
        X = np.asarray([[[ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
                         [ 0.,  0.,  1.,  1.,  1.,  2.,  0.],
                         [ 0.,  0.,  2.,  1.,  1.,  2.,  0.],
                         [ 0.,  0.,  2.,  2.,  2.,  2.,  0.],
                         [ 0.,  1.,  1.,  1.,  0.,  0.,  0.],
                         [ 0.,  1.,  1.,  0.,  2.,  2.,  0.],
                         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.]],

                        [[ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
                         [ 0.,  0.,  0.,  0.,  2.,  1.,  0.],
                         [ 0.,  1.,  0.,  0.,  0.,  1.,  0.],
                         [ 0.,  1.,  2.,  2.,  0.,  1.,  0.],
                         [ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
                         [ 0.,  0.,  0.,  0.,  1.,  1.,  0.],
                         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.]],

                        [[ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
                         [ 0.,  2.,  2.,  0.,  1.,  2.,  0.],
                         [ 0.,  1.,  1.,  0.,  2.,  2.,  0.],
                         [ 0.,  2.,  0.,  1.,  1.,  2.,  0.],
                         [ 0.,  0.,  0.,  0.,  1.,  0.,  0.],
                         [ 0.,  1.,  2.,  0.,  0.,  0.,  0.],
                         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.]]]).astype(np.float32)
        # expectedX = np.asarray([,])
        gX = clarray.to_device(clplatf.qs[0], X)
        gcol, ev = conv.im2col(gX, (3,3), 1, 2)
        ev.wait()
        col = gcol.get()

        self.assertEqual(len(col.shape), 2)
        self.assertTrue(np.allclose(col.shape, (27, 9)))

        print >>sys.stderr, "\nX:", X
        print >>sys.stderr, "\ncol:", col