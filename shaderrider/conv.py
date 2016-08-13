"""Convolution things"""

import numpy as np
import pyopencl as cl
from pyopencl import array as clarray

from shaderrider import clplatf


def im2col(img, rec_field, n_filters, stride=1, zero_pad=0, wait_for=None):
    """

    :type stride: int
    :type zero_pad: int
    """
    dtype = 'float' if img.dtype == np.float32 else 'double'
    q = clplatf.qs[0]
    d1,h1,w1 = img.shape

    kh, kw = rec_field
    out_h = kh*kw*d1

    w2 = (w1 - kw + 2 * zero_pad) / stride + 1
    h2 = (h1 - kh + 2 * zero_pad) / stride + 1
    # TODO check if w2 or h2 is not int and raise something or zeropad...
    d2 = n_filters
    out_w = w2*h2
    # alloc output
    col = clarray.Array(q, (out_h, out_w), img.dtype)     # FIXME ne ove dimenzije (konvolucionog outa) nego matrice...

    prg = cl.Program(clplatf.ctx, """
    __kernel void im2col_k(__global %(dtype)s *img,
                            int h,
                            int w,
                            int out_h,
                            int out_w,
                            int kh,
                            int kw,
                            int stride,
                            int padding,
                            __global %(dtype)s *out) {
        int gid = get_global_id(0);
        int ch = gid / (kh * kw * out_h * out_w);
        int out_x = gid %% out_w;
        int out_y = gid / out_w %% out_h;
        int kx = gid / (out_h*out_w) %% kw;
        int ky = gid / (kw*out_h*out_w) %% kh;
        int in_x = kx + out_x*stride - padding;
        int in_y = ky + out_y*stride - padding;
        if (in_x >= 0 && in_x < w && in_y >= 0 && in_y < h) {
            out[gid] = img[(in_y + h*ch)*w + in_x];
        } else {
            out[gid] = 0;
        }
    }
    """ % locals()).build()

    evt = prg.im2col_k(q, (out_h*out_w,), None,
                       img.data,
                       np.int32(h1),
                       np.int32(w1),
                       np.int32(out_h),
                       np.int32(out_w),
                       np.int32(kh),
                       np.int32(kw),
                       np.int32(stride),
                       np.int32(zero_pad),
                       col.data,
                       wait_for=wait_for)

    return col, evt


def col2im(col, ):
    pass
