"""Convolution things"""

import numpy as np
import pyopencl as cl
from pyopencl import array as clarray
from pyopencl.elementwise import ElementwiseKernel

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
    # TODO cache the kernel, this creates new one every time
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
        int x = gid %% w + pw;

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
    # TODO cache the kernel, this creates new one every time
    evt = prg.col2im(q, (n*c*h*w,), None,
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


def sum_by_axis(q, a, axis):
    dtype = 'float' if a.dtype == np.float32 else 'double'
    element_size = 4 if a.dtype == np.float32 else 8
    prg = cl.Program(clplatf.ctx, r"""
        __kernel void ax_sum(__global float *gdata, int n, int dim_n, int sn,
                int dstride, int dskip, int estride,
                __global float *output) {
            int gid = get_global_id(0);
            float sum = 0.0;
            int offset = (gid!=0)*((gid*dstride) % sn + (gid/dim_n)*dskip);
            for (int i=0; i<dim_n; i++) {
                sum += gdata[i*estride + offset];
            }
            output[gid] = sum;
        }
        """).build()
    ax_sum_k = prg.ax_sum
    axes = reversed(sorted(list(axis)))
    in_a = a
    out_shape = a.shape
    pev = None  # previous event
    for dim in axes:
        # TODO calc out dims and kernel dim/stride params
        out_shape.pop(dim)
        dn = in_a.shape[dim]
        n = in_a.size
        sn = None
        estride = in_a.strides[dim]/element_size
        out_a = clarray.empty(q, out_shape, in_a.dtype)
        launch_shape = int(np.prod(in_a.shape[:dim])*(np.prod(in_a.shape[dim+1:])
                                                      if dim < in_a.ndim-1 else 1))
        ev = ax_sum_k(q, (launch_shape,), None,
                      in_a.data,
                      np.int32(n),
                      np.int32(dn),
                      np.int32(sn),
                      np.int32(dstride),
                      np.int32(dskip),
                      np.int32(estride),
                      out_a.data,
                      wait_for=pev)
        pev = [ev]
    return out_a, pev


def bcast_add(q, A, b, out=None):
    """Bradcast add b vector to 3d tensor A"""
    dtype = 'float' if A.dtype == np.float32 else 'double'
    element_size = 4 if A.dtype == np.float32 else 8

    badd = ElementwiseKernel(clplatf.ctx,
                             'float *X, float *y, float *out, int nb, int c, int w, int h',
                             'out[i] = X[i] + y[(i/w/h)%c]',
                             "badd")

    # TODO assert A and b shapes
    if out is None:
        # TODO zero padding maybe?
        out = clarray.empty_like(A)

    evt = badd(A, b, out,
               np.int32(A.shape[0]),
               np.int32(A.shape[1]),
               np.int32(A.shape[2]),
               np.int32(A.shape[3]))

    return out, evt


def bgrads_sum(q, gY, out):
    dtype = 'float' if gY.dtype == np.float32 else 'double'
    # element_size = 4 if gY.dtype == np.float32 else 8
    n, c, h, w = gY.shape
    step3_gridsize = 0
    if n>=256:
        step3_gridsize = 256
    elif n>=128:
        step3_gridsize = 128
    elif n>=64:
        step3_gridsize = 64
    else:
        raise ValueError

    prg = cl.Program(clplatf.ctx, """
        __kernel void sumLastD(__global %(dtype)s *gdata, int w, __global %(dtype)s *output) {
            int gid = get_global_id(0);
            int offset = gid*w;
            %(dtype)s sum = 0;
            for (int i=0; i<w; i++) {
                sum += gdata[i + offset];
            }
            output[gid] = sum;
        }

        __kernel void rsum(__global %(dtype)s *gdata, __global %(dtype)s *odata,
                           int n, int c, __local %(dtype)s *sdata) {
            int gid = get_global_id(0);
            int tid = get_local_id(0);
            int blockSize = get_local_size(0)/2;
            int gridSize = get_global_size(0);
            int i;

            for (i=0; i<c; i++)
                sdata[c*tid+i] = 0;

            i = gid;
            while (i < n) {
                for (unsigned short d=0; d<c; d++)
                    sdata[c*tid+d] += gdata[c*i+d] + gdata[c*(i+blockSize)+d];
                i += gridSize;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            for (unsigned int s=blockSize/2; s>0; s>>=1) {
                if (tid < s)
                    for (unsigned short d=0; d<c; d++)
                        sdata[c*tid+d] += sdata[c*(tid + s)+d];
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (tid == 0) {
                for (unsigned short d=0; d<c; d++)
                    gdata[c*gid+d] = sdata[d];
            }
            if (gid == 0) {
                for (unsigned short d=0; d<c; d++)
                    odata[d] += sdata[d];
            }
        }
        """ % locals()).build()

    kstep1 = prg.sumLastD
    kstep3 = prg.rsum
    temp1 = clarray.zeros(q, (n, c, h), gY.dtype)
    temp2 = clarray.zeros(q, (n, c), gY.dtype)
    ev1 = kstep1(q, (n*c*h,), None, gY.data, np.int32(w), temp1.data)
    ev2 = kstep1(q, (n*c,), None, temp1.data, np.int32(h), temp2.data,
                 wait_for=[ev1])

    locMemSize = step3_gridsize*c*(4 if gY.dtype==np.float32 else 8)
    ev3 =  kstep3(q, (step3_gridsize,), (step3_gridsize,),
                  temp2.data, out.data, np.int32(n), np.int32(c),
                  cl.LocalMemory(locMemSize),
                  wait_for=[ev2])
    return out, ev3
