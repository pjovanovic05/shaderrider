"""Big test for reverse gradient on Conv2d."""
import unittest
import numpy as np
import pyopencl as cl
from pyopencl import array as clarray

from shaderrider import clplatf as pl
from shaderrider import expr
from shaderrider import operators as op


def _col2im_cpu(col, sy, sx, ph, pw, h, w):
    n, c, kh, kw, out_h, out_w = col.shape

    img = np.zeros((n, c, h + 2 * ph + sy - 1, w + 2 * pw + sx - 1),
                   dtype=col.dtype)
    for i in xrange(kh):
        i_lim = i + sy * out_h
        for j in xrange(kw):
            j_lim = j + sx * out_w
            img[:, :, i:i_lim:sy, j:j_lim:sx] += col[:, :, i, j, :, :]

    return img[:, :, ph:h + ph, pw:w + pw]


@unittest.skip('unfinished test case for convolution operator')
class Conv2dTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pl.init_cl(1)

    def _backward_cpu(self, inputs, grad_outputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        gy = grad_outputs[0]
        h, w = x.shape[2:]

        gW = np.tensordot(
            gy, self.col, ((0, 2, 3), (0, 4, 5))).astype(W.dtype, copy=False)
        gcol = np.tensordot(W, gy, (0, 1)).astype(x.dtype, copy=False)
        gcol = np.rollaxis(gcol, 3)
        gx = _col2im_cpu(gcol, self.sy, self.sx, self.ph, self.pw, h, w)

        if b is None:
            return gx, gW
        else:
            gb = gy.sum(axis=(0, 2, 3))
            return gx, gW, gb

    def test_rev_grad(self):
        pass
