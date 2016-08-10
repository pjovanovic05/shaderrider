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

    # TODO check dims and call gemm, gemv or ger...
    if M == 1:
        if N == 1:
            # TODO vector dot product
            scratch = clarray.empty_like(a, queue=q)
            ev = clblaswrap.dot(q, a, b, out, scratch)
        else:
            # TODO gemv where vector is on the left - will need some transpositions
            pass
    elif K == 1:
        # TODO outer product
        ev = clblaswrap.ger(q, out, a, b)
        pass
    elif M > 1:
        if N == 1:
            # TODO standard gemv
            ev = clblaswrap.gemv(q, a, b, out)
            pass
        else:
            # TODO gemm finally!
            ev = clblaswrap.gemm(q, a, b, out)
            pass

    # TODO batch gemm goes to a different op

    return out, ev  # TODO return the event too

