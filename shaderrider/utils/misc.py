"""Miscelaneous low level clarray operations."""
import numpy as np
import pyopencl as cl
from pyopencl.tools import dtype_to_ctype
from pyopencl import array as clarray
from shaderrider import clplatf
from shaderrider.aux import clblaswrap


# FIXME Trik sa gemv-om iz mlp primera, moze samo da sumira 2d nizove...
def sum(q, a, axis=None, out=None, keepdims=False):
    if axis is None or a.ndim <= 1:
        out_shape = (1,)*a.ndim if keepdims else ()
        return clarray.sum(a).reshape(out_shape)

    if axis < 0:
        axis += 2
    if axis > 1:
        raise ValueError('invalid axis')

    if a.flags.c_contiguous:
        m, n = a.shape
        lda = a.shape[1]
        transA = False if axis == 0 else True
        sum_axis, out_axis = (m, n) if axis == 0 else (n, m)
    else:
        n, m = a.shape
        lda = a.shape[0]
        transA = True if axis == 0 else False
        sum_axis, out_axis = (n, m) if axis == 0 else (m, n)

    ones = clarray.empty(q, (sum_axis,), a.dtype).fill(1)
    if keepdims:
        out_shape = (1, out_axis) if axis == 0 else (out_axis, 1)
    else:
        out_shape = (out_axis,)

    if out is None:
        out = clarray.zeros(q, out_shape, a.dtype)
    else:
        assert out.dtype == a.dtype
        assert out.size >= out_axis

    if a.dtype == np.float32:
        gemv = clblaswrap.sgemv
    elif a.dtype == np.float64:
        gemv = clblaswrap.dgemv
    else:
        raise TypeError('Unsupported array type: %s' % str(a.dtype))

    alpha = 1.0
    beta = 0.0

    ev = gemv(q, transA, m, n, alpha, a, lda, ones, 1, beta, out, 1)
    ev.wait()

    return out


_cmp_by_axis_kernel_template = """
    __kernel void _max_by_rows(__global %(dtype)s *A,
                            __global %(dtype)s *out, __global int *ids,
                            int w, int h,
                            __local %(dtype)s *maxs, __local int *maxids) {
        int tid = get_local_id(0);
        int workgroup_sz = get_local_size(0);
        int blockid = get_global_id(0)/workgroup_sz;
        unsigned int curr_idx = 0;
        %(dtype)s curr_max = %(init_val)s;
        %(dtype)s curr_val = 0;

        //find local max
        for (int i=tid; i<w; i+=workgroup_sz) {
            curr_val = A[blockid * w + i];
            if (curr_val %(cmp_op)s curr_max) {
                curr_max = curr_val;
                curr_idx = i;
            }
        }

        maxs[tid] = curr_max;
        maxids[tid] = curr_idx;
        barrier(CLK_LOCAL_MEM_FENCE);

        //find global max
        if (tid == 0) {
            curr_max = %(init_val)s;
            curr_idx = 0;
            for (int i=0; i<min(workgroup_sz, w); i++)
                if (maxs[i] %(cmp_op)s curr_max) {
                    curr_max = maxs[i];
                    curr_idx = maxids[i];
                }
            out[blockid] = curr_max;
            ids[blockid] = curr_idx;
        }
    }

    __kernel void _max_by_cols(__global %(dtype)s *A,
                            __global %(dtype)s *out, __global int *ids,
                            int w, int h,
                            __local %(dtype)s *maxs, __local int *maxids) {
        int tid = get_local_id(0);
        int workgroup_sz = get_local_size(0);
        int blockid = get_global_id(0)/workgroup_sz;
        unsigned int curr_idx = 0;
        %(dtype)s curr_max = %(init_val)s;
        %(dtype)s curr_val = 0;

        //find thread max
        for (int i=tid; i<h; i+=workgroup_sz) {
            curr_val = A[blockid + i*w];
            if (curr_val %(cmp_op)s curr_max) {
                curr_max = curr_val;
                curr_idx = i;
            }
        }

        maxs[tid] = curr_max;
        maxids[tid] = curr_idx;
        barrier(CLK_LOCAL_MEM_FENCE);

        //find workgroup max
        if (tid == 0) {
            curr_max = %(init_val)s;
            curr_idx = 0;
            for (int i=0; i<min(workgroup_sz, h); i++)
                if (maxs[i] %(cmp_op)s curr_max) {
                    curr_max = maxs[i];
                    curr_idx = maxids[i];
                }
            out[blockid] = curr_max;
            ids[blockid] = curr_idx;
        }
    }"""


def max(q, a, axis=None, keepdims=False):
    assert a.ndim < 3

    if axis is None or a.ndim <= 1:
        out_shape = (1,)*a.ndim
        return clarray.max(a).reshape(out_shape)
    elif axis < 0:
        axis += 2
    assert axis in (0, 1)

    # TODO generate & cache kernel elsewhere
    prg = cl.Program(clplatf.ctx,
                     _cmp_by_axis_kernel_template % {
                            'cmp_op': '>',
                            'dtype': dtype_to_ctype(a.dtype),
                            'init_val': str(np.finfo(a.dtype).min)
                        }).build()
    col_max = prg._max_by_cols
    row_max = prg._max_by_rows
    # TODO calculate workgroup size given the array and axis
    element_size = 4 if a.dtype == np.float32 else 8
    n, m = a.shape if a.flags.c_contiguous else (a.shape[1], a.shape[0])
    if (axis == 0 and a.flags.c_contiguous) or (axis == 1 and a.flags.f_contiguous):
        if keepdims:
            out_shape = (1, m) if axis == 0 else (m, 1)
        else:
            out_shape = (m,)
        maxes = clarray.empty(q, out_shape, dtype=a.dtype)
        indices = clarray.empty(q, out_shape, dtype=np.int32)
        ev = col_max(q, (m*64,), (64,), a.data, maxes.data, indices.data, np.int32(m), np.int32(n),
                     cl.LocalMemory(64*element_size), cl.LocalMemory(64*4))
    else:
        if keepdims:
            out_shape = (1, n) if axis == 0 else (n, 1)
        else:
            out_shape = (n,)
        maxes = clarray.empty(q, out_shape, dtype=a.dtype)
        indices = clarray.empty(q, out_shape, dtype=np.int32)
        ev = row_max(q, (n*64,), (64,), a.data, maxes.data, indices.data, np.int32(m), np.int32(n),
                     cl.LocalMemory(64*element_size), cl.LocalMemory(64*4))
    if ev is not None:
        ev.wait()
    return maxes, indices


# TODO max se ponasa cudno... konsultuj https://scikit-cuda.readthedocs.io/en/latest/_modules/skcuda/misc.html#sum
# TODO napisi unittest
