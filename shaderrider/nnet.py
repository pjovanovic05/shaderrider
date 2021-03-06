"""Miscelaneous neural net related operations."""
import numpy as np
import pyopencl as cl
from pyopencl.tools import dtype_to_ctype
from pyopencl import array as clarray
from shaderrider import clplatf


def argmax(q, A, dim, out=None):
    dtype = dtype_to_ctype(A.dtype)
    nstride = A.strides[dim]/4
    n = A.shape[dim]
    mstride = A.strides[dim-1]/4
    m = A.shape[dim-1]
    prg = cl.Program(clplatf.ctx, """
        __kernel void argmaxk(__global %(dtype)s *A, int n, int m, int nstride,
                              int mstride, __global %(dtype)s *output) {
            int gid = get_global_id(0);
            int i = gid*mstride;
            int maxj = 0;
            %(dtype)s maxv = A[i];
            for (int j=1; j<n; j++) {
                //%(dtype)s d = maxv - A[i+j*nstride];
                //maxj = ceil(fmax(d, 0))*maxj + ceil(fmax(-d, 0))*j;
                if (A[i+j*nstride]>maxv) {
                    maxv = A[i+j*nstride];
                    maxj = j;
                }
            }
            output[gid] = maxj;
        }
    """ % locals()).build()

    argmaxk = prg.argmaxk

    if out is None:
        out = clarray.empty(q, A.shape[:dim], dtype=A.dtype)

    ev = argmaxk(q, out.shape, None,
                 A.data,
                 np.int32(n),
                 np.int32(m),
                 np.int32(nstride),
                 np.int32(mstride),
                 out.data)
    return out, ev
