"""
Linear algebra functions.

Uses clBLAS as optimization when possible
"""

import numpy as np
import pyopencl as cl
from pyopencl import array as clarray

from shaderrider.aux import clblaswrap
from shaderrider import clplatf


def dot(queue, a, b, out=None, wait_for=None):
    if len(a.shape) > 2:
        raise TypeError, 'A is not a matrix'
    if len(b.shape) > 2:
        raise TypeError, 'B is not a matrix'

    M, K = a.shape if len(a.shape) == 2 else a.shape[0],1
    N = b.shape[1] if len(b.shape) == 2 else 1
    ev = None

    if out is None:
        out = clarray.empty(queue, (M, N), a.dtype)

    if M == 1:
        if N == 1:
            # vector dot product
            scratch = clarray.empty_like(a, queue=queue)
            ev = clblaswrap.dot(queue, b, a, out, scratch)
        else:
            # gemv where vector is on the left - will need some transpositions
            ev = clblaswrap.gemv(queue, a, b, out)
    elif K == 1:
        # outer product
        ev = clblaswrap.ger(queue, out, a, b)
    elif M > 1:
        if N == 1:
            # standard gemv
            ev = clblaswrap.gemv(queue, a, b, out)
        else:
            # gemm finally!
            ev = clblaswrap.gemm(queue, a, b, out)

    # TODO batch gemm goes to a different op

    return out, ev

