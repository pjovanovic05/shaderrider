"""Convolution things"""

import numpy as np
import pyopencl as cl
from pyopencl import array as clarray

from shaderrider import clplatf


def im2col_old(img, rec_field, n_filters, stride=1, zero_pad=0, wait_for=None):
    """

    :type stride: int
    :type zero_pad: int
    """
    dtype = 'float' if img.dtype == np.float32 else 'double'
    q = clplatf.qs[0]
    d1,h1,w1 = img.shape

    kh, kw = rec_field
    out_h = kh*kw*d1

    w2 = (w1 - kw + 2 * zero_pad) // stride + 1
    h2 = (h1 - kh + 2 * zero_pad) // stride + 1
    # TODO check if w2 or h2 is not int and raise something or zeropad...
    out_w = w2*h2
    # alloc output
    col = clarray.Array(q, (out_h, out_w), img.dtype)

    prg = cl.Program(clplatf.ctx, """
    __kernel void im2col_k(__global %(dtype)s *img,
                            int h,
                            int w,
                            int o_h,
                            int o_w,
                            int kh,
                            int kw,
                            int stride,
                            int padding,
                            __global %(dtype)s *out) {
        int gid = get_global_id(0);
        int out_w = o_w * o_h;
        int out_h = kw * kh;
        int out_x = gid %% out_w;
        int out_y = gid / out_w %% out_h;
        int kx = out_y %% kw;
        int ky = (out_y / kh) %% kh;
        int ch = gid / (kh * kw * o_h * o_w);

        int in_x = kx + (out_x %% o_w)*stride - padding;
        int in_y = ky + (out_x / o_w)*stride - padding;

        if (in_x >= 0 && in_x < w && in_y >= 0 && in_y < h) {
            out[gid] = img[(h * ch + in_y) * w + in_x];
        } else {
            out[gid] = 0;
        }
    }
    """ % locals()).build()

    evt = prg.im2col_k(q, (out_h*out_w,), None,
                       img.data,
                       np.int32(h1),
                       np.int32(w1),
                       np.int32(h2),
                       np.int32(w2),
                       np.int32(kh),
                       np.int32(kw),
                       np.int32(stride),
                       np.int32(zero_pad),
                       col.data,
                       wait_for=wait_for)

    return col, evt


def get_conv_outsize(size, k, s, p, cover_all=False):
    if cover_all:
        return (size + p * 2 - k + s - 1) // s + 1
    else:
        return (size + p * 2 -k) // s + 1


def im2col(q, img, kh, kw, sy, sx, ph, pw, cover_all=False):
    # TODO better dtype conversion
    dtype = 'float' if img.dtype == np.float32 else 'double'
    n, c, h, w = img.shape
    out_h = get_conv_outsize(h, kh, sy, ph, cover_all)
    out_w = get_conv_outsize(w, kw, sx, pw, cover_all)

    col = clarray.empty(q, (n, c, kh, kw, out_h, out_w), img.dtype)
    prg = cl.Program(clplatf.ctx, """
    __kernel void im2col(__global %(dtype)s *img, int h, int w, int out_h, int out_w,
                         int kh, int kw, int sy, int sx, int ph, int pw,
                         __global %(dtype)s *col) {
        int gid = get_global_id(0);
        int c0 = gid / (kh * kw * out_h * out_w);
        int ky = gid / (kw * out_h * out_w) %% kh;
        int kx = gid / (out_h * out_w) %% kw;
        int out_y = gid / out_w %% out_h;
        int out_x = gid %% out_w;
        int in_y = ky + out_y * sy - ph;
        int in_x = kx + out_x * sx - pw;

        if (in_y>=0 && in_y<h && in_x>=0 && in_x<w)
            col[gid] = img[in_x + w * (in_y + h * c0)];
        else
            col[gid] = 0;
    }
    """ % locals()).build()

    evt = prg.im2col(q, (n*c*kh*kw*out_h*out_w,), None,
                     img.data,
                     np.int32(h),
                     np.int32(w),
                     np.int32(out_h),
                     np.int32(out_w),
                     np.int32(kh),
                     np.int32(kw),
                     np.int32(sy),
                     np.int32(sx),
                     np.int32(ph),
                     np.int32(pw),
                     col.data)

    return col, evt


def col2im(q, col, sy, sx, ph, pw, h, w, wait_for=None):
    # TODO better dtype conversion
    dtype = 'float' if col.dtype == np.float32 else 'double'
    n, c, kh, kw, out_h, out_w = col.shape
    img = clarray.empty(q, (n, c, h, w), col.dtype)

    prg = cl.Program(clplatf.ctx, """
    __kernel void col2im(__global %(dtype)s *col, int h, int w, int out_h, int out_w,
                         int kh, int kw, int sy, int sx, int ph, int pw,
                         __global %(dtype)s *img) {
        int gid = get_global_id(0);
        int c0 = gid / (h*w);
        int y = gid / w %% h + ph;
        int x = gid %% w + ph;

        int out_y_0 = max(0, (y-kh+sy)/sy);
        int out_y_1 = min(out_h, (y+sy)/sy);
        int out_x_0 = max(0, (x-kw+sx)/sx);
        int out_x_1 = min(out_w, (x+sx)/sx);

        %(dtype)s val = 0;
        for (int out_y=out_y_0; out_y<out_y_1; out_y++) {
            int ky = y-out_y*sy;
            for (int out_x=out_x_0; out_x<out_x_1; out_x++) {
                int kx = x-out_x*sx;
                int k = out_y+out_h*(kx+kw*(ky+kh*c0));
                val += col[out_x+out_w*k];
            }
        }
        img[gid] = val;
    }
    """ % locals()).build()

    evt = prg.col2im(q, (h*w,), None,
                       col.data,
                       np.int32(h),
                       np.int32(w),
                       np.int32(out_h),
                       np.int32(out_w),
                       np.int32(kh),
                       np.int32(kw),
                       np.int32(sy),
                       np.int32(sx),
                       np.int32(ph),
                       np.int32(pw),
                       img.data,
                       wait_for=wait_for)
    return img, evt
