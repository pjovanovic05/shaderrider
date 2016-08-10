"""
Linear algebra functions.

Uses clBLAS as optimization when possible
"""

import numpy as np
import pyopencl as cl
from pyopencl import array as clarray

from shaderrider.aux import clblaswrap
from shaderrider import clplatf


def dot(a, b, out=None, wait_for=None):
    if len(a.shape) > 2:
        raise TypeError, 'A is not a matrix'
    if len(b.shape) > 2:
        raise TypeError, 'B is not a matrix'

    M, K = a.shape
    N = b.shape[1]

    q = clplatf.qs[0]   # FIXME
    if out is None:
        out = clarray.empty(q, (M, N), a.dtype)

    if M == 1:
        if N == 1:
            # vector dot product
            scratch = clarray.empty_like(a, queue=q)
            ev = clblaswrap.dot(q, b, a, out, scratch)
        else:
            # gemv where vector is on the left - will need some transpositions
            ev = clblaswrap.gemv(q, a, b, out)
    elif K == 1:
        # outer product
        ev = clblaswrap.ger(q, out, a, b)
    elif M > 1:
        if N == 1:
            # standard gemv
            ev = clblaswrap.gemv(q, a, b, out)
        else:
            # gemm finally!
            ev = clblaswrap.gemm(q, a, b, out)

    # TODO batch gemm goes to a different op

    return out, ev

