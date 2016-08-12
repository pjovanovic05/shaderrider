"""Convolution things"""

import pyopencl as cl
from pyopencl import array as clarray

from shaderrider import clplatf


def im2col(img, rec_field, n_filters, stride=1, zero_pad=0):
    q = clplatf.qs[0]
    d1,h1,w1 = img.shape
    w2 = (w1 - rec_field + 2 * zero_pad) / stride + 1
    h2 = (h1 - rec_field + 2 * zero_pad) / stride + 1
    # TODO check if w2 or h2 is not int and raise something or zeropad...
    d2 = n_filters
    # alloc output
    col = clarray.Array(q, (w2, h2, d2), img.dtype)     # FIXME ne ove dimenzije (konvolucionog outa) nego matrice...

    prg = cl.Program(clplatf.ctx, """
    __kernel void im2col_k(__global %(dtype)s im, __global %(dtype)s dst) {
        //TODO
        int gid = get_global_id();

    }
    """).build()

    return col  # TODO event too probably...

def col2im():
    pass
