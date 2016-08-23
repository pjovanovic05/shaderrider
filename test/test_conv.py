import unittest
import numpy as np
import pyopencl as cl
from pyopencl import array as clarray

from shaderrider import clplatf
from shaderrider import conv


class ConvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        clplatf.init_cl(1)
        # TODO setup X and gX

    @classmethod
    def tearDownClass(cls):
        pass

    def test_im2col_old(self):
        X = np.asarray([[[0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 1., 1., 1., 2., 0.],
                         [0., 0., 2., 1., 1., 2., 0.],
                         [0., 0., 2., 2., 2., 2., 0.],
                         [0., 1., 1., 1., 0., 0., 0.],
                         [0., 1., 1., 0., 2., 2., 0.],
                         [0., 0., 0., 0., 0., 0., 0.]],

                        [[0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 2., 1., 0.],
                         [0., 1., 0., 0., 0., 1., 0.],
                         [0., 1., 2., 2., 0., 1., 0.],
                         [0., 1., 1., 1., 1., 1., 0.],
                         [0., 0., 0., 0., 1., 1., 0.],
                         [0., 0., 0., 0., 0., 0., 0.]],

                        [[0., 0., 0., 0., 0., 0., 0.],
                         [0., 2., 2., 0., 1., 2., 0.],
                         [0., 1., 1., 0., 2., 2., 0.],
                         [0., 2., 0., 1., 1., 2., 0.],
                         [0., 0., 0., 0., 1., 0., 0.],
                         [0., 1., 2., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0.]]]).astype(np.float32)
        expectedX = np.asarray([[0., 0., 0., 0., 2., 1., 0., 1., 0.],
                                [0., 0., 0., 0., 1., 2., 1., 1., 0.],
                                [0., 0., 0., 2., 1., 0., 1., 0., 0.],
                                [0., 1., 1., 0., 2., 2., 0., 1., 2.],
                                [0., 1., 2., 0., 2., 2., 1., 0., 2.],
                                [1., 1., 0., 2., 2., 0., 1., 2., 0.],
                                [0., 2., 1., 0., 1., 0., 0., 0., 0.],
                                [0., 1., 2., 1., 1., 0., 0., 0., 0.],
                                [2., 1., 0., 1., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 1., 1.],
                                [0., 0., 0., 1., 0., 1., 1., 1., 1.],
                                [0., 0., 0., 0., 0., 0., 1., 1., 0.],
                                [0., 0., 2., 0., 2., 0., 0., 0., 1.],
                                [0., 0., 1., 1., 2., 1., 0., 0., 1.],
                                [0., 2., 0., 2., 0., 0., 0., 1., 0.],
                                [0., 0., 0., 0., 1., 1., 0., 0., 0.],
                                [1., 0., 1., 1., 1., 1., 0., 0., 0.],
                                [0., 0., 0., 1., 1., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 1., 2., 0., 0., 1.],
                                [0., 0., 0., 1., 0., 2., 0., 0., 0.],
                                [0., 0., 0., 1., 2., 0., 0., 1., 0.],
                                [0., 2., 1., 0., 0., 1., 0., 2., 0.],
                                [2., 0., 2., 2., 1., 2., 1., 0., 0.],
                                [2., 1., 0., 0., 1., 0., 2., 0., 0.],
                                [0., 1., 2., 0., 0., 1., 0., 0., 0.],
                                [1., 0., 2., 0., 0., 0., 0., 0., 0.],
                                [1., 2., 0., 0., 1., 0., 0., 0., 0.]]).astype(np.float32)
        gX = clarray.to_device(clplatf.qs[0], X)
        gcol, ev = conv.im2col_old(gX, (3, 3), 1, 2)
        ev.wait()
        col = gcol.get()

        self.assertEqual(len(col.shape), 2)
        self.assertTrue(np.allclose(col.shape, (27, 9)))
        self.assertTrue(np.allclose(expectedX, col))

    def test_im2col(self):
        X = np.asarray([[[[0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 1., 1., 1., 2., 0.],
                         [0., 0., 2., 1., 1., 2., 0.],
                         [0., 0., 2., 2., 2., 2., 0.],
                         [0., 1., 1., 1., 0., 0., 0.],
                         [0., 1., 1., 0., 2., 2., 0.],
                         [0., 0., 0., 0., 0., 0., 0.]],

                        [[0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 2., 1., 0.],
                         [0., 1., 0., 0., 0., 1., 0.],
                         [0., 1., 2., 2., 0., 1., 0.],
                         [0., 1., 1., 1., 1., 1., 0.],
                         [0., 0., 0., 0., 1., 1., 0.],
                         [0., 0., 0., 0., 0., 0., 0.]],

                        [[0., 0., 0., 0., 0., 0., 0.],
                         [0., 2., 2., 0., 1., 2., 0.],
                         [0., 1., 1., 0., 2., 2., 0.],
                         [0., 2., 0., 1., 1., 2., 0.],
                         [0., 0., 0., 0., 1., 0., 0.],
                         [0., 1., 2., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0.]]]]).astype(np.float32)
        expectedX = np.asarray([[[0., 0., 0., 0., 2., 1., 0., 1., 0.],
                                [0., 0., 0., 0., 1., 2., 1., 1., 0.],
                                [0., 0., 0., 2., 1., 0., 1., 0., 0.],
                                [0., 1., 1., 0., 2., 2., 0., 1., 2.],
                                [0., 1., 2., 0., 2., 2., 1., 0., 2.],
                                [1., 1., 0., 2., 2., 0., 1., 2., 0.],
                                [0., 2., 1., 0., 1., 0., 0., 0., 0.],
                                [0., 1., 2., 1., 1., 0., 0., 0., 0.],
                                [2., 1., 0., 1., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 1., 1.],
                                [0., 0., 0., 1., 0., 1., 1., 1., 1.],
                                [0., 0., 0., 0., 0., 0., 1., 1., 0.],
                                [0., 0., 2., 0., 2., 0., 0., 0., 1.],
                                [0., 0., 1., 1., 2., 1., 0., 0., 1.],
                                [0., 2., 0., 2., 0., 0., 0., 1., 0.],
                                [0., 0., 0., 0., 1., 1., 0., 0., 0.],
                                [1., 0., 1., 1., 1., 1., 0., 0., 0.],
                                [0., 0., 0., 1., 1., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 1., 2., 0., 0., 1.],
                                [0., 0., 0., 1., 0., 2., 0., 0., 0.],
                                [0., 0., 0., 1., 2., 0., 0., 1., 0.],
                                [0., 2., 1., 0., 0., 1., 0., 2., 0.],
                                [2., 0., 2., 2., 1., 2., 1., 0., 0.],
                                [2., 1., 0., 0., 1., 0., 2., 0., 0.],
                                [0., 1., 2., 0., 0., 1., 0., 0., 0.],
                                [1., 0., 2., 0., 0., 0., 0., 0., 0.],
                                [1., 2., 0., 0., 1., 0., 0., 0., 0.]]]).astype(np.float32)
        gX = clarray.to_device(clplatf.qs[0], X)
        gcol, ev = conv.im2col(clplatf.qs[0], gX, 3, 3, 2, 2, 0, 0)
        ev.wait()
        col = gcol.reshape((1,-1,27,9)).get()

        self.assertTrue(np.allclose(expectedX, col))

    def test_col2im(self):
        pass

    def test_sum_by_axis(self):
        pass

    def test_bcast_add(self):
        pass

    def test_bgrads_sum(self):
        pass
