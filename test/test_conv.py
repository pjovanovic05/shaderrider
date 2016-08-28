import unittest
import numpy as np
import pyopencl as cl
from pyopencl import array as clarray

from shaderrider import clplatf
from shaderrider import conv

# import sys


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
        col = gcol.reshape((1, -1, 27, 9)).get()

        self.assertTrue(np.allclose(expectedX, col))

    def test_col2im(self):
        q = clplatf.qs[0]

        X = np.arange(5*3*7*7).astype(np.float32)
        X.shape = (5, 3, 7, 7)
        kh = 3
        kw = 3
        sx = 2
        sy = 2
        ph = 0
        pw = 0
        h = 7
        w = 7
        gX = clarray.to_device(q, X)
        gcolX, ev = conv.im2col(q, gX, kh, kw, sy, sx, ph, pw)
        ev.wait()

        # colX = gcolX.get()
        # print >>sys.stderr, '\ncolX: (%s)\n' % str(colX.shape), colX

        gX2, ev2 = conv.col2im(q, gcolX, sy, sx, ph, pw, h, w)

        ev2.wait()

        X2 = gX2.get()
        # print >>sys.stderr, '\nX2 (%s):\n' % str(X2.shape), X2/X
        self.assertTrue(np.allclose(X.shape, X2.shape))

    # def test_sum_by_axis(self):
    #     pass

    def test_bcast_add(self):
        q = clplatf.qs[0]
        A = np.arange(30*200*200).astype(np.float32)
        A.shape = (10, 3, 200, 200)
        b = np.arange(3).astype(np.float32)
        gA = clarray.to_device(q, A)
        gb = clarray.to_device(q, b)

        gout, ev = conv.bcast_add(q, gA, gb)
        ev.wait()

        out = gout.get()
        self.assertTrue(np.allclose(out, A+b[:, None, None]))

    def test_bgrads_sum(self):
        q = clplatf.qs[0]
        X = np.arange(512*3*200*200).astype(np.float32)
        X.shape = (512, 3, 200, 200)
        expected = X.sum(axis=(0, 2, 3))

        gX = clarray.to_device(q, X)
        gout = clarray.empty(q, (3,), dtype=np.float32)

        _, evt = conv.bgrads_sum(q, gX, gout)
        evt.wait()
        out = gout.get()
        self.assertTrue(np.allclose(expected, out))
