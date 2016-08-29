import unittest
import numpy as np
import pyopencl as cl
from pyopencl import array as clarray

from shaderrider import clplatf as pl
from shaderrider import expr
from shaderrider import operators as op

# import sys


class OperatorsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pl.init_cl(1)

    def test_dot_eval(self):
        x = expr.Variable('x')
        y = expr.Variable('y')
        dotprod = op.Dot(x, y)

        valuation = pl.valuation()
        nx = np.random.uniform(0, 1, (10,)).astype(np.float32)
        ny = np.random.uniform(0, 1, (10,)).astype(np.float32)
        valuation['x'] = nx
        valuation['y'] = ny
        gd = dotprod.evaluate(valuation)
        d = gd.get()
        self.assertTrue(np.allclose(d, np.dot(nx, ny)))
        # TODO tests for different dimensions cases...

    def test_dot_fwdgrad(self):
        x = expr.Variable('x')
        y = expr.Variable('y')
        dotprod = op.Dot(x, y)

        valuation = pl.valuation()
        nx = np.random.uniform(0, 1, (10,)).astype(np.float32)
        ny = np.random.uniform(0, 1, (10,)).astype(np.float32)
        valuation['x'] = nx
        valuation['y'] = ny
        xw = clarray.zeros(pl.qs[0], (10,), dtype=np.float32) + 1.0
        yw = clarray.zeros(pl.qs[0], (10,), dtype=np.float32)

        gddot = dotprod.fwd_grad({'x': xw, 'y': yw}, valuation)
        ddot = gddot.get()
        # print >>sys.stderr, '\nddot: ', ddot

    def test_dot_revgrad(self):
        x = expr.Variable('x')
        y = expr.Variable('y')
        dotprod = op.Dot(x, y)

        valuation = pl.valuation()
        nx = np.random.uniform(0, 1, (10,)).astype(np.float32)
        ny = np.random.uniform(0, 1, (10,)).astype(np.float32)
        valuation['x'] = nx
        valuation['y'] = ny

        grad = dotprod.rev_grad(valuation)
        dx = grad['x'].get()
        dy = grad['y'].get()
        self.assertTrue(np.allclose(dx, ny))
        self.assertTrue(np.allclose(dy, nx))

    def test_conv2d_eval(self):
        img = expr.Variable('img')
        k = expr.Variable('k')
        b = expr.Variable('b')
        convolution = op.Conv2d(img, k, b, strides=(2, 2), zero_padding=(0, 0))

        valuation = pl.valuation()
        nimg = np.asarray([[[[0., 0., 0., 0., 0., 0., 0.],
                             [0., 2., 1., 0., 2., 2., 0.],
                             [0., 2., 2., 1., 1., 2., 0.],
                             [0., 2., 0., 0., 2., 1., 0.],
                             [0., 0., 2., 0., 1., 0., 0.],
                             [0., 1., 1., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0.]],

                            [[0., 0., 0., 0., 0., 0., 0.],
                             [0., 2., 0., 2., 0., 1., 0.],
                             [0., 2., 2., 1., 0., 2., 0.],
                             [0., 2., 1., 0., 0., 1., 0.],
                             [0., 2., 0., 1., 0., 0., 0.],
                             [0., 1., 1., 0., 2., 2., 0.],
                             [0., 0., 0., 0., 0., 0., 0.]],

                            [[0., 0., 0., 0., 0., 0., 0.],
                             [0., 2., 0., 0., 2., 1., 0.],
                             [0., 2., 2., 2., 2., 0., 0.],
                             [0., 1., 1., 2., 1., 0., 0.],
                             [0., 2., 1., 2., 1., 0., 0.],
                             [0., 1., 0., 0., 0., 2., 0.],
                             [0., 0., 0., 0., 0., 0., 0.]]]]).astype(np.float32)
        nk = np.asarray([[
                [[-1, 1, 1],
                 [1, 0, 0],
                 [1, 0, 0]],
                [[1, -1, 1],
                 [1, -1, -1],
                 [1, 1, 1]],
                [[0, 0, 0],
                 [0, -1, 1],
                 [0, 1, -1]]
            ],
            [
                [[0, 0, -1],
                 [0, 1, 0],
                 [0, 0, -1]],
                [[-1, -1, 1],
                 [-1, 1, -1],
                 [1, 1, 0]],
                [[0, 1, -1],
                 [1, 0, 0],
                 [0, -1, 0]]
            ]]).astype(np.float32)
        nb = np.asarray([1, 0]).astype(np.float32)
        expected = np.asarray([[[[1, 7, 4],
                                 [5, 6, 2],
                                 [-2, -1, -2]],
                                [[2, 2, 7],
                                 [-1, -6, 1],
                                 [-2, -4, 0]]]]).astype(np.float32)

        valuation['img'] = nimg
        valuation['k'] = nk
        valuation['b'] = nb
        ret = convolution.evaluate(valuation)
        nret = ret.get()
        self.assertTrue(np.allclose(expected, nret))

    def test_conv2d_revgrad(self):
        pass
