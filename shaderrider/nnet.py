"""Miscelaneous neural net related operations."""
import numpy as np
import pyopencl as cl
from pyopencl import array as clarray
from shaderrider import clplatf


def argmax(q, A, dim, out=None):
    dtype = 'float' if A.dtype == np.float32 else 'double'
    nstride = A.strides[dim]
    n = A.shape[dim]
    mstride = A.strides[dim-1]
    m = A.shape[dim-1]
    prg = cl.Program(clplatf.ctx, """
        __kernel void argmaxk(__global %(dtype)s *A, int n, int m, int nstride,
                              int mstride, __global %(dtype)s *output) {
            //TODO
            int gid = get_global_id(0);
            int i = gid*mstride;
            int maxj = 0;
            %(dtype)s maxv = A[i];
            for (int j=1; j<n; j++) {
                %(dtype)s d = maxv - A[i+j*nstride];
                maxj = fmax(d, 0)*maxj + ceil(fmax(-d, 0))*j;
            }
            out[gid] = maxj;
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
